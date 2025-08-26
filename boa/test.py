import logging
import time
import json
import argparse
import contextlib
from typing import List, Optional
import numpy as np
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm

import omegaconf
import torch
from lightning.pytorch import seed_everything

from torch.utils.data import Subset
from torch.utils.data import DataLoader
from scdp.scdp.common.pyg.dataloader import Collater
from scdp.scdp.data.dataset import LmdbDataset
from scdp.scdp.data.datamodule import worker_init_fn
from boa.data.basis_info import BasisInfo, build_molecule_np
from boa.data.of_data import AddMessagePassingMatrix, AddRadiusEdgeIndex, OFBatch, OFData, ToTorch
from scdp.scdp.model.utils import get_nmape
from scdp.scdp.model.module import ChgLightningModule, move_batch_to_dtype

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("highest")

class OfBatchCollater(Collater):
    def __init__(self, follow_batch, exclude_keys, basis_info):
        super().__init__(follow_batch, exclude_keys)
        self.basis_info = basis_info

    def __call__(self, batch):
        of_data_list = []
         
        for x in batch:
            mol = build_molecule_np(charges = x.atom_types,
                        positions = x.coords, basis = self.basis_info.basis_dict)
            of_data = ToTorch()(OFData.minimal_sample_from_mol(mol, self.basis_info))
            of_data = AddRadiusEdgeIndex(radius=6.0)(of_data)
            of_data = AddMessagePassingMatrix(self.basis_info, add_edge_matrices=True)(of_data)
            of_data.message_edge_index = of_data.edge_index.clone()
            of_data.message_edge_matrices = of_data.edge_matrices.clone()
            of_data = AddRadiusEdgeIndex(radius=3.0)(of_data)
            of_data = AddMessagePassingMatrix(self.basis_info, add_edge_matrices=True)(of_data)
            of_data_list.append(of_data)

        # batch = [x.sample_probe(n_probe=min(self.n_probe, x.n_probe)) for x in batch]
        batch = super().__call__(batch)
        of_batch = OFBatch.from_data_list(of_data_list, ["coeffs", "atomic_numbers"])
        batch.of_batch = of_batch
        return batch

def get_data_probe_chunk(input_data, indices):
    data = deepcopy(input_data)
    data['chg_labels'] = input_data.chg_labels[indices]
    data['probe_coords'] = input_data.probe_coords[indices]
    data['n_probe'] = len(indices)
    data['sampled'] = True
    return data

def get_probe_chunks(n_probes, max_n_probe_per_pass):
    batch_size = len(n_probes)
    n_per_pass = []
    probes_to_process = []
    n_probes = torch.clone(n_probes)
    probe_indices = torch.arange(n_probes.sum(), device=n_probes.device)
    
    n_pass = ((n_probes).sum() / max_n_probe_per_pass).ceil().long()
    pass_start_index = 0
    pass_end_index = 0
    for _ in range(n_pass):
        current_pass = torch.zeros(batch_size, dtype=torch.long, device=n_probes.device)
        current_load = 0
        
        for i in range(batch_size):
            if n_probes[i] == 0:
                continue    
            max_points_for_job = (max_n_probe_per_pass - current_load)
            points_to_process = torch.min(torch.tensor([n_probes[i], max_points_for_job]))
            
            current_load += points_to_process
            current_pass[i] = points_to_process
            pass_end_index += int(points_to_process)
            
            # in-place modification done last
            n_probes[i] -= points_to_process
            
            if current_load >= max_n_probe_per_pass:
                break
        n_per_pass.append(current_pass)
        probes_to_process.append(probe_indices[pass_start_index:pass_end_index])
        pass_start_index = pass_end_index
        pass_end_index = pass_start_index
    
    return n_pass, n_per_pass, probes_to_process

class OFBatchDataLoader(DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = [None],
        exclude_keys: Optional[List[str]] = [None],
        basis_info: BasisInfo = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.basis_info = basis_info

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=OfBatchCollater(
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
                basis_info=basis_info,
            ),
            **kwargs,
        )

def main(
    ckpt_path,
    data_path,
    split_file,
    tag='test',
    max_n_graphs=10000,
    batch_size=4,
    max_n_probe=500000,
    use_last=False,
    ):
    seed_everything(42)
    ckpt_path = Path(ckpt_path)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                filename=ckpt_path / f'eval_{tag}.log',
                filemode='w')
        
    cfg = omegaconf.OmegaConf.load(ckpt_path / "config.yaml")

    # load checkpoint
    storage_dir: str = Path(ckpt_path)
    if (storage_dir / 'last.ckpt').exists() and use_last:
        ckpt = storage_dir / 'last.ckpt'
    else:
        ckpts = list(storage_dir.glob("*epoch*.ckpt"))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split("-")[0].split("=")[1]) for ckpt in ckpts]
            )
            ckpt_ix = ckpt_epochs.argsort()[-1]
            ckpt = str(ckpts[ckpt_ix])
        else:
            raise FileNotFoundError(f"No checkpoint found in <{ckpt_path}>.")
    
    # pylint: disable=E1120
    pylogger.info(f"loaded checkpoint: {ckpt}")
    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt).to('cuda')
    model.eval()
    model.ema.copy_to(model.parameters())
    
    # set up data loader
    dataset = LmdbDataset(data_path)
    with open(split_file, "r") as fp:
        splits = json.load(fp)
    test_dataset = Subset(dataset, splits['test'])
    test_loader = OFBatchDataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        worker_init_fn=worker_init_fn,
        basis_info=model.basis_info,
    )

    # Better load balancing possible.
    pylogger.info("starting testing.")    
    with(torch.no_grad(),
         contextlib.ExitStack() as context_stack, 
         open(storage_dir / f'nmape_{tag}.txt', 'w') as f
        ):
                
        prog = context_stack.enter_context(tqdm(total=min(max_n_graphs, len(test_dataset)), disable=None))
        display_bar = context_stack.enter_context(
                tqdm(
                    bar_format=""
                    if prog.disable  # prog.ncols doesn't exist if disabled
                    else ("{desc:." + str(prog.ncols) + "}"),
                    disable=None,
                )
            )

        curr_time = time.time()
        
        idx = 0
        all_nmapes = []
        for batch in test_loader:
            batch = batch.to('cuda')
            batch = move_batch_to_dtype(batch, model.dtype)
            coeffs, expo_scaling, edge_index = model.predict_coeffs(batch)
            n_pass, n_per_pass, probes_to_process = get_probe_chunks(batch.n_probe, max_n_probe)
            all_preds = []
            for i_pass in range(n_pass):
                n_probe = n_per_pass[i_pass]
                probe_idx = probes_to_process[i_pass]
                probe_coords = batch.probe_coords[probe_idx]
                pred = model.orbital_inference(batch, coeffs, expo_scaling, n_probe, probe_coords, edge_index)
                all_preds.append(pred)
            all_preds = torch.cat(all_preds, dim=0)
            nmape = get_nmape(
                all_preds, batch.chg_labels, 
                torch.arange(len(batch), device=all_preds.device).repeat_interleave(batch.n_probe)
            ).cpu().numpy().tolist()
            all_nmapes.extend(nmape)
            
            for item in nmape:
                f.write(f'{item}\n')
            f.flush()
            prog.update(batch.num_graphs)
            display_bar.set_description_str(f"nmape: {np.mean(all_nmapes):.4f} ± {np.std(all_nmapes):.4f}")
            idx += batch.num_graphs
            if idx >= max_n_graphs:
                break
        #save the nmape results ckpt_path / f'nmape_{tag}.txt'
        torch.save(all_nmapes, ckpt_path / f'nmape_{tag}.pt')
        
        elapsed_time = time.time() - curr_time
        pylogger.info(f"elapsed time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint folder.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        help="Path to the split file.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="test",
        help="split to use",
    )
    parser.add_argument(
        "--max_n_graphs",
        type=int,
        default=10000,
        help="max number of data points to do inference for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max_n_probe",
        type=int,
        default=180000,
        help="max number of probes to process in one pass.",
    )
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use the last checkpoint.",
    )
    args: argparse.Namespace = parser.parse_args()
    main(args.ckpt_path, args.data_path, args.split_file, args.tag, 
         args.max_n_graphs, args.batch_size, args.max_n_probe, args.use_last)