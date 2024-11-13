import torch


class SimpleEvaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        print("Resetting Evaluator")
        self.mious = []

    def calculate_miou(self, seg_bev_e, seg_bev_g, valid_bev_g):
        
        seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum(dim=[1,2])
        union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum(dim=[1,2])
        iou = (intersection/(1e-4 + union)).mean()

        # batch mean
        self.mious.append(iou.item())

    
    def update(self, pred, gt, valid=None):

        self.calculate_miou(pred, gt, valid)

        last_miou = self.mious[-1]

        results = self.get_results(reset=False)

        results['last_mIoU'] = last_miou

        return results

    def get_results(self, reset=True):

        miou = sum(self.mious) / len(self.mious)

        if reset:
            self.reset()
        
        return {'mIoU' : miou}
    
class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        print("Resetting Evaluator")
        self.mious = []
    
    def calculate_miou(self, pred, gt, valid):
            
        pred_round = (pred > 0.5).float()  
        intersection = (pred_round*gt*valid).sum(dim=[1,2])
        union = ((pred_round+gt)*valid).clamp(0,1).sum(dim=[1,2])
        iou = (intersection/(1e-4 + union)).mean()

        # batch mean
        self.mious.append(iou.item())

    def update(self, pred, gt, valid=None):

        self.calculate_miou(pred, gt, valid)

        last_miou = self.mious[-1]

        results = self.get_results(reset=False)

        results['last_mIoU'] = last_miou

        return results
    
    def get_results(self, reset=True):
            
        miou = sum(self.mious) / len(self.mious)

        if reset:
            self.reset()
        
        return {'mIoU' : miou}
    
# class Evaluator:
#     def __init__(self):
        
#         self.thresholds = [0.5, 0.9, 0.95, 0.99, 0.999]
#         self.reset()

#     def reset(self):
#         print("Resetting Evaluator")
#         #self.mious = []
#         for threshold in self.thresholds:
#             setattr(self, f'mious_{threshold}', [])

#     def calculate_miou(self, pred, gt, valid):

#         for threshold in self.thresholds:
#             pred_round = (pred > threshold).float()  
#             intersection = (pred_round*gt*valid).sum(dim=[1,2])
#             union = ((pred_round+gt)*valid).clamp(0,1).sum(dim=[1,2])
#             iou = (intersection/(1e-4 + union)).mean()

#             # batch mean
#             getattr(self, f'mious_{threshold}').append(iou.item())
        
#         # pred_round = (pred > 0.5).float()  
#         # if valid is None:
#         #     valid = torch.ones_like(gt)
            
#         # intersection = (pred_round*gt*valid).sum(dim=[1,2])
#         # union = ((pred_round+gt)*valid).clamp(0,1).sum(dim=[1,2])
#         # iou = (intersection/(1e-4 + union)).mean()

#         # # batch mean
#         # self.mious.append(iou.item())

    
#     def update(self, pred, gt, valid=None):

#         self.calculate_miou(pred, gt, valid)

#         results_ = {}
#         for threshold in self.thresholds:
#             last_miou = getattr(self, f'mious_{threshold}')[-1]
#             results_[f'last_mIoU_{threshold}'] = last_miou

#         results = self.get_results(reset=False)

#         results.update(results_)
        
#         # last_miou = self.mious[-1]
#         # results = self.get_results(reset=False)
#         # results['last_mIoU'] = last_miou

#         return results

#     def get_results(self, reset=True):

#         results = {}

#         for threshold in self.thresholds:
#             miou = sum(getattr(self, f'mious_{threshold}')) / len(getattr(self, f'mious_{threshold}'))
#             results[f'mIoU_{threshold}'] = miou
        
#         #miou = sum(self.mious) / len(self.mious)

#         if reset:
#             self.reset()
        
#         return results
    
