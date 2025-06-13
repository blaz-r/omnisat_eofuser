from pathlib import Path
from typing import Any

from PIL import Image
import torch
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from torchmetrics import Metric


class Visualiser(Callback):
    def __init__(
        self,
        res_path: Path,
        add_vis_keys: list = None,
        criterion: Metric | None = None,
        criterion_limit: float = 1,
    ):
        self.save_path = Path(res_path)
        self.save_path.mkdir(exist_ok=True)

        vis_keys = [
            ((0), "aerial", "Drone"),
            ((1), "label", "GT"),
            ((2), "pred_mask", "Pred. mask"),
            ((3), "pred", "Pred. map"),
        ]
        if add_vis_keys is not None:
            vis_keys.extend(add_vis_keys)

        self.vis_keys = vis_keys

        self.criterion = criterion
        self.criterion_limit = criterion_limit

    def visualize(self, batch):
        num = len(batch["name"])
        for s_idx in range(num):
            img_name = Path(batch["name"][s_idx]).name

            pred_map = batch["pred"][s_idx].detach().cpu()
            pred_mask = torch.argmax(pred_map, dim=0).detach().cpu().squeeze()
            # take only class 1
            pred_map = pred_map[1]
            gt_mask = batch["label"][s_idx].detach().cpu()

            plot_current = True
            val = None
            if self.criterion is not None:
                raise NotImplementedError("Not implemented with validity mask.")
                val = self.criterion(pred_mask, torch.where(gt_mask > 0.5, 1, 0))

                plot_current = val < self.criterion_limit

            if not plot_current:
                continue
            # plot only if criterion indicates a poor sample

            pred_map = pred_map.squeeze()#.numpy()
            if isinstance(self.vis_keys[0][0], int):
                di = 1
                dj = max(p for (p), _, _ in self.vis_keys) + 1
            else:
                di = max(p for (p, _), _, _ in self.vis_keys) + 1
                dj = max(p for (_, p), _, _ in self.vis_keys) + 1

            fig, plots = plt.subplots(di, dj, figsize=(9, 6))
            for s_plt in plots.flatten():
                s_plt.axis("off")
            fig.tight_layout()

            if val is not None:
                if di == 1:
                    plots[2].title.set_text(f"F1: {val:.3f}")
                else:
                    plots[0][2].title.set_text(f"F1: {val:.3f}")

            for pidx, key, name in self.vis_keys:
                item = batch.get(key, None)

                if item is not None:
                    item = item[s_idx].detach().cpu().squeeze()

                if key == "aerial":
                    # image
                    item = item.permute(1, 2, 0)
                    item = torch.clip(item / 255, 0, 1)

                if key == "pred_mask":
                    plots[pidx].imshow(pred_mask, vmax=1, vmin=0, cmap="gray")
                    plots[pidx].title.set_text("Pred. mask")
                elif key == "pred":
                    plots[pidx].imshow(pred_map, vmax=1, vmin=0)
                    plots[pidx].title.set_text("Pred. map")

                    pred_maps_dir = self.save_path / "pred_map"
                    pred_maps_dir.mkdir(exist_ok=True, parents=True)

                    # save as png for lossless mask
                    Image.fromarray(pred_mask.float().numpy() * 255).convert("RGB").save(str(pred_maps_dir / f"{img_name}.png"))
                else:
                    plots[pidx].imshow(item)
                    plots[pidx].title.set_text(name)

            fig.tight_layout()
            plt.savefig(self.save_path / f"{img_name}.jpg", bbox_inches="tight")

            plt.close("all")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.visualize(batch=outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
