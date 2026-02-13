import numpy as np
import torch


def SEDI(predicted_values, true_values, percentile, var_sample_counts):


    percentile = percentile.cpu().numpy() if torch.is_tensor(percentile) else percentile
    pred = predicted_values.numpy() if torch.is_tensor(predicted_values) else predicted_values
    true = true_values.numpy() if torch.is_tensor(true_values) else true_values
    
    num_percentile = percentile.shape[-1]  
    num_vars = len(var_sample_counts) 
    num_pairs = num_percentile // 2  
    output_length = true.shape[1]  
    
    pred_events = np.zeros((num_pairs, num_vars))
    gt_events = np.zeros((num_pairs, num_vars))
    
    start_idx = 0
    for var_idx in range(num_vars):
        var_count = var_sample_counts[var_idx]  
        end_idx = start_idx + var_count  
        
        var_true = true[start_idx:end_idx, :]
        var_pred = pred[start_idx:end_idx, :]
        

        for i in range(num_pairs):
           
            low_thresh = percentile[0, 0, var_idx, i]  
            high_thresh = percentile[0, 0, var_idx, num_percentile - 1 - i]  
            
            gt_low = (var_true < low_thresh).flatten()
            gt_high = (var_true > high_thresh).flatten()
            gt_total = np.sum(gt_low) + np.sum(gt_high)
            
            pred_low_hit = np.sum(np.logical_and((var_pred < low_thresh).flatten(), gt_low))
            pred_high_hit = np.sum(np.logical_and((var_pred > high_thresh).flatten(), gt_high))
            pred_total = pred_low_hit + pred_high_hit
            
            pred_events[i, var_idx] = pred_total
            gt_events[i, var_idx] = gt_total

        start_idx = end_idx

    return pred_events, gt_events

class MultiMetricsCalculator:
    def __init__(self, num_vars=6):
        self.num_vars = num_vars  
        self.SEDI_pred = np.zeros((4, num_vars))  
        self.SEDI_gt = np.zeros((4, num_vars))    
        self.count = 0  

    def update(self, predicted_values, true_values, percentile, var_sample_counts):
        pred = predicted_values.cpu().numpy() if torch.is_tensor(predicted_values) else predicted_values
        true = true_values.cpu().numpy() if torch.is_tensor(true_values) else true_values
        
        pred_events, gt_events = SEDI(pred, true, percentile, var_sample_counts)
        self.SEDI_pred += pred_events
        self.SEDI_gt += gt_events
        self.count += 1

    def get_metrics(self):

        if self.count == 0:
            return np.zeros((4, self.num_vars))  #
        SEDI = np.zeros((4, self.num_vars))
        for i in range(4):  
            for j in range(self.num_vars):  
                if self.SEDI_gt[i, j] > 0:
                    SEDI[i, j] = self.SEDI_pred[i, j] / self.SEDI_gt[i, j]
                else:
                    SEDI[i, j] = 0.0
        
        return SEDI
    
class ProbabilisticSEDI:
    @staticmethod
    def calculate(pred_samples, true_values, percentile, var_sample_counts):
        if pred_samples.dim() != 3:
            pred_samples = pred_samples.squeeze(-1)
            true_values = true_values.squeeze(-1)

        percentile = percentile.cpu().numpy() if torch.is_tensor(percentile) else percentile
        pred = pred_samples.cpu().numpy() if torch.is_tensor(pred_samples) else pred_samples
        true = true_values.cpu().numpy() if torch.is_tensor(true_values) else true_values
        
        num_percentile = percentile.shape[-1]
        num_vars = len(var_sample_counts)
        num_pairs = num_percentile // 2
        
        pred_events = np.zeros((num_pairs, num_vars))
        gt_events = np.zeros((num_pairs, num_vars))
        
        start_idx = 0
        for var_idx in range(num_vars):
            var_count = var_sample_counts[var_idx]
            end_idx = start_idx + var_count
            
            var_pred_samples = pred[:, start_idx:end_idx, :]
            var_true = true[start_idx:end_idx, :]
            var_percentiles = percentile[0, 0, var_idx, :]
            
            for pair_idx in range(num_pairs):
                low_thresh = var_percentiles[pair_idx]
                high_thresh = var_percentiles[num_percentile - 1 - pair_idx]
                
                gt_low_mask = (var_true < low_thresh).flatten()
                gt_high_mask = (var_true > high_thresh).flatten()
                gt_total = np.sum(gt_low_mask) + np.sum(gt_high_mask)
                gt_events[pair_idx, var_idx] = gt_total
                
                sample_min = var_pred_samples.min(axis=0)
                sample_max = var_pred_samples.max(axis=0)
                sample_median = np.median(var_pred_samples, axis=0)
                
                global_low_pct = np.percentile(var_pred_samples, 10)
                global_high_pct = np.percentile(var_pred_samples, 90)
                
                low_mask = (sample_min < global_low_pct)
                high_mask = (sample_max > global_high_pct)
                normal_mask = ~(low_mask | high_mask)

                var_converted_pred = np.zeros_like(sample_median)
                var_converted_pred[low_mask] = sample_min[low_mask]
                var_converted_pred[high_mask] = sample_max[high_mask]
                var_converted_pred[normal_mask] = sample_median[normal_mask]

                flat_pred = var_converted_pred.flatten()
                pred_low_hit = np.sum(np.logical_and((flat_pred < low_thresh), gt_low_mask))
                pred_high_hit = np.sum(np.logical_and((flat_pred > high_thresh), gt_high_mask))
                pred_events[pair_idx, var_idx] = pred_low_hit + pred_high_hit
            
            start_idx = end_idx

        return pred_events, gt_events

class ProbabilisticMultiMetricsCalculator:
    def __init__(self, num_vars=6):
        self.num_vars = num_vars
        self.SEDI_pred = np.zeros((4, num_vars))
        self.SEDI_gt = np.zeros((4, num_vars))
        self.count = 0

    def update(self, predicted_samples, true_values, percentile, var_sample_counts):
        pred_events, gt_events = ProbabilisticSEDI.calculate(
            predicted_samples, true_values, percentile, var_sample_counts
        )
        self.SEDI_pred += pred_events
        self.SEDI_gt += gt_events
        self.count += 1

    def get_metrics(self):
        if self.count == 0:
            return np.zeros((4, self.num_vars))
        
        sedi_matrix = np.zeros((4, self.num_vars))
        for i in range(4):
            for j in range(self.num_vars):
                if self.SEDI_gt[i, j] > 0:
                    sedi_matrix[i, j] = self.SEDI_pred[i, j] / self.SEDI_gt[i, j]
        return sedi_matrix