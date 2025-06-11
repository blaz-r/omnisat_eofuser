from torchmetrics import Metric
from torchmetrics import F1Score
import torch
import os

class MetricsMonoModal(Metric):
    """
    Computes the micro, macro and weighted F1 Score for multi label classification
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False
    ):
        super().__init__()
        self.modality = modalities[0]
        self.get_classes = get_classes
        self.f1 = F1Score(task="multilabel", average = "none", num_labels=num_classes)
        self.f1_micro = F1Score(task="multilabel", average = "micro", num_labels=num_classes)
        self.f1_weighted = F1Score(task="multilabel", average = "weighted", num_labels=num_classes)
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        self.f1(pred, gt['label'])
        self.f1_micro(pred, gt['label'])
        self.f1_weighted(pred, gt['label'])
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred.cpu()[i].numpy())

    def compute(self):
        if self.get_classes:
            f1 = self.f1.compute()
            out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
            for i in range(len(f1)):
                out['_'.join(['F1_classe', str(i)])] = f1[i]
            return out
        f1 = self.f1.compute()
        out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
        if self.save_results:
            out['results'] = self.results
            return out
        return out

class MetricsMultiModal(Metric):
    """
    Computes the micro, macro and weighted F1 Score for multi label classification with UT&T model
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file
        get_modalities (bool): if True returns the F1 Score for sub branch of UT&T
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_modalities: bool = False,
        get_classes: bool = False,
    ):
        super().__init__()
        self.modalities = modalities
        self.get_modalities = get_modalities
        self.get_classes = get_classes
        self.f1 = F1Score(task="multilabel", average = "none", num_labels=num_classes - 2)
        self.f1_micro = F1Score(task="multilabel", average = "micro", num_labels=num_classes - 2)
        self.f1_weighted = F1Score(task="multilabel", average = "weighted", num_labels=num_classes - 2)
        if self.get_modalities:
            self.f1_m = {}
            for m in self.modalities:
                self.f1_m[m] = F1Score(task="multilabel", average = "none", num_labels=num_classes).cpu()
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        self.f1(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        self.f1_micro(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        self.f1_weighted(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        if self.get_modalities:
            for m in self.modalities:
                self.f1_m[m](pred[m][:,1:-1].cpu(), gt['label'][:,1:-1].cpu())
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred[self.modalities[0]].cpu()[i].numpy())

    def compute(self):
        f1 = self.f1.compute()
        out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
        if self.save_results:
            out['results'] = self.results
        if self.get_modalities:
            for m in self.modalities:
                out['_'.join(['F1_Score', m])] = self.f1_m[m].compute()
        if self.get_classes:
            for i in range(len(f1)):
                out['_'.join(['F1_classe', str(i)])] = f1[i]
        return out

class NoMetrics(Metric):
    """
    Computes no metrics or saves a batch of reconstruction to visualise them
    Args:
        save_reconstructs (bool): if True saves a batch of reconstructions
        modalities (list): list of modalities used
        save_dir (str): where to save reconstructions
    """

    def __init__(
        self,
        save_reconstructs: bool = False,
        modalities: list = [],
        save_dir: str = '',
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_recons = save_reconstructs
        self.modalities = modalities
        if self.save_recons:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.saves = {}
            for modality in self.modalities:
                self.saves[modality] = []
                self.saves['_'.join(['gt', modality])] = []

    def update(self, pred, gt):
        if self.save_recons:
            recons, _ = pred
            for modality in self.modalities:
                if modality == 'aerial':
                    preds = recons['_'.join(['reconstruct', modality])]
                    preds  = preds.permute(0, 2, 1 ,3, 4)
                    preds  = preds.reshape(preds.shape[0],4,6,6,50,50)
                    preds = preds.permute(0, 1, 2, 4, 3, 5)
                    preds = preds.reshape(preds.shape[0],4,300,300)
                    target = gt[modality][:, :, :300, :300]
                else:
                    preds, mask = recons['_'.join(['reconstruct', modality])]
                    preds = preds.view(-1, preds.shape[1], 6, 6)
                    target = gt[modality][mask[:, 0], mask[:, 1]]
                indice = torch.randint(0, len(preds), (1,)).item()
                self.saves[modality].append(preds[indice])
                self.saves['_'.join(['gt', modality])].append(target[indice])

    def compute(self):
        if self.save_recons:
            for key in self.saves.keys():
                for i, tensor in enumerate(self.saves[key]):
                    torch.save(tensor.cpu(), self.save_dir + key + str(i) + ".pt")
        return {}

class MetricsContrastif(Metric):
    """
    Computes metrics for contrastive. Given embeddings for all tokens, we compute the cosine similarity matrix.
    The metric computed is the accuracy of the M -1 minimum distances of each line (except diagonal of course)
    being the same token across other modalities with M the number of modalities.
    Args:
        modalities (list): list of modalities used
    """

    def __init__(
        self,
        modalities: list = [],
    ):
        super().__init__()
        self.modalities = modalities
        self.n_k = len(self.modalities)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        for i in range(len(modalities)):
            self.add_state(modalities[i], default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits):
        size = len(logits) // self.n_k
        labels = torch.arange(size).unsqueeze(1)
        labels = torch.cat([labels + i * len(labels) for i in range(self.n_k)], dim=1)
        labels = torch.cat([labels for _ in range(self.n_k)]).to(logits.device)
        for i in range(self.n_k):
            _, top_indices = torch.topk(logits[i * size:(i + 1) * size], k=self.n_k, dim=1, largest=True)
            self.__dict__[self.modalities[i]] += (torch.sum(torch.tensor([top_indices[i, j] in labels[i]
                                                for i in range(top_indices.size(0)) for j in range(self.n_k)])) - len(top_indices)) / (self.n_k - 1)
        self.count += len(logits)

    def compute(self):
        dict = {}
        for i in range(len(self.modalities)):
            dict['_'.join(['acc', self.modalities[i]])] = self.__dict__[self.modalities[i]] / self.count
        return dict


class SegPangaea(Metric):
    """
    SegPangaea is a class for evaluating segmentation models using a confusion matrix approach.

    Attributes:
        num_classes (int): Number of classes in the segmentation task
        ignore_index (int): Index value to ignore when computing metrics
        confusion_matrix (torch.Tensor): Matrix of shape (num_classes, num_classes) to store predictions

    Methods:
        update(pred, gt):
            Updates the confusion matrix with new predictions and ground truth.
            Args:
                pred (torch.Tensor): Model predictions
                gt (dict): Dictionary containing ground truth labels under 'label' key

        compute():
            Computes various metrics from the accumulated confusion matrix.
            Returns:
                dict: Dictionary containing the following metrics:
                    - mIoU: Mean Intersection over Union across all classes
                    - mF1: Mean F1 score across all classes
                    - mAcc: Mean pixel accuracy
    """

    def __init__(self, num_classes, ignore_index):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, pred, gt):
        label = gt['label'].flatten(1).long()
        pred = torch.argmax(pred, dim=1).flatten(1)
        valid_mask = label != self.ignore_index
        pred, target = pred[valid_mask], label[valid_mask]
        count = torch.bincount(
            (pred * self.num_classes + target), minlength=self.num_classes ** 2
        )
        self.confusion_matrix = self.confusion_matrix.to(pred.device)
        self.confusion_matrix += count.view(self.num_classes, self.num_classes)

    def compute(self):
        # Calculate IoU for each class
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6))

        # Calculate precision and recall for each class
        precision = intersection / (self.confusion_matrix.sum(dim=0) + 1e-6)
        recall = intersection / (self.confusion_matrix.sum(dim=1) + 1e-6)

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        miou = iou.mean().item()
        mf1 = f1.mean().item()
        macc = (intersection.sum() / (self.confusion_matrix.sum() + 1e-6)).item()

        # Convert metrics to CPU and to Python scalars
        iou = iou.cpu()
        f1 = f1.cpu()
        precision = precision.cpu()
        recall = recall.cpu()

        # Prepare the metrics dictionary
        metrics = {
            "mIoU": miou,
            "mF1": mf1,
            "mAcc": macc,
        }

        return metrics