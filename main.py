import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import os

# Define the ResNet-101 model for feature extraction
class ResNet101(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=pretrained)
        
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-2])
    
    def forward(self, x):
        x = self.model(x)
        return x

# Define the Region Proposal Network (RPN)
class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        super(RPN, self).__init__()
        
        self.anchor_base = self.generate_anchor_base(ratios, anchor_scales)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, len(ratios) * len(anchor_scales) * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, len(ratios) * len(anchor_scales) * 4, 1, 1, 0)

        # Define loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.smoothL1Loss = nn.SmoothL1Loss()

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        
        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)
        rpn_scores = self.score(x)

        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)

        anchor = self._enumerate_shifted_anchor(np.array(self.anchor_base), h, w)
        
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    
    def generate_anchor_base(self, ratios, anchor_scales):
        # Generate anchor base windows by enumerating aspect ratios and scales
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                h = anchor_scales[j] * np.sqrt(ratios[i])
                w = anchor_scales[j] * np.sqrt(1. / ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = - h / 2.
                anchor_base[index, 1] = - w / 2.
                anchor_base[index, 2] = h / 2.
                anchor_base[index, 3] = w / 2.
        return anchor_base
    
    def _enumerate_shifted_anchor(self, anchor_base, h, w):
        # Enumerate all shifted anchors
        shift_x = np.arange(0, w) * 16
        shift_y = np.arange(0, h) * 16
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

        A = anchor_base.shape[0]
        K = shift.shape[0]
        anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
        anchor = anchor.reshape((K * A, 4)).astype(np.float32)
        return anchor
    
    def proposal_layer(self, rpn_locs, rpn_fg_scores, anchor, img_size, scale):
        # Convert rpn locs to actual proposal positions
        n_anchor = anchor.shape[0]
        tx = rpn_locs[:, 0::4]
        ty = rpn_locs[:, 1::4]
        tw = rpn_locs[:, 2::4]
        th = rpn_locs[:, 3::4]
        
        x = tx.detach().numpy() * 16 + anchor[:, 0]
        y = ty.detach().numpy() * 16 + anchor[:, 1]
        w = np.exp(tw.detach().numpy()) * (anchor[:, 2] - anchor[:, 0])
        h = np.exp(th.detach().numpy()) * (anchor[:, 3] - anchor[:, 1])
        
        proposal = np.stack((x, y, x + w, y + h), axis=-1).astype(np.float32)
        
        # Filter out proposals outside the image
        proposal[:, [0, 2]] = np.clip(proposal[:, [0, 2]], 0, img_size[1])
        proposal[:, [1, 3]] = np.clip(proposal[:, [1, 3]], 0, img_size[0])
        
        # Remove proposals with small areas
        keep = self._filter_small_boxes(proposal)
        proposal = proposal[keep, :]
        rpn_fg_scores = rpn_fg_scores[keep]
        
        # Apply NMS to remaining proposals
        keep = self._nms(proposal, rpn_fg_scores)
        proposal = proposal[keep, :]
        
        return proposal
    
    def _filter_small_boxes(self, boxes, min_size=16):
        # Remove boxes with any side smaller than min_size
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    def _nms(self, proposal, rpn_fg_scores):
        # Non-Maximum Suppression
        x1 = proposal[:, 0]
        y1 = proposal[:, 1]
        x2 = proposal[:, 2]
        y2 = proposal[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(rpn_fg_scores)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= 0.5)[0]
            order = order[inds + 1]

        return keep

# Define the Faster R-CNN model
class FasterRCNN(nn.Module):
    def __init__(self, n_classes, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        super(FasterRCNN, self).__init__()
        
        self.extractor = ResNet101()
        self.rpn = RPN(2048, 512, ratios, anchor_scales)
        self.head = RoIHead(n_classes, 7, 2048, 512)

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

# Define the RoI Head
class RoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, in_channels, fc_channels):
        super(RoIHead, self).__init__()

        self.n_classes = n_classes
        self.roi_size = roi_size
        self.spatial_scale = 1.0 / 16.0
        
        self.roi_align = RoIAlign(self.roi_size, self.roi_size, self.spatial_scale)
        self.fc1 = nn.Linear(in_channels * self.roi_size * self.roi_size, fc_channels)
        self.fc2 = nn.Linear(fc_channels, fc_channels)
        self.cls_loc = nn.Linear(fc_channels, n_classes * 4)
        self.score = nn.Linear(fc_channels, n_classes)
        
        self._initialize_weights()

    def forward(self, x, rois, roi_indices):
        # Extract proposal feature vectors using RoI Align
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi_align(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)

        fc1 = F.relu(self.fc1(pool))
        fc2 = F.relu(self.fc2(fc1))
        roi_cls_locs = self.cls_loc(fc2)
        roi_scores = self.score(fc2)

        return roi_cls_locs, roi_scores

    def _initialize_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Define the RoI Align layer
class RoIAlign(nn.Module):
    def __init__(self, out_h, out_w, spatial_scale):
        super(RoIAlign, self).__init__()
        
        self.out_h = out_h
        self.out_w = out_w
        self.spatial_scale = spatial_scale

    def forward(self, feature_maps, rois):
        batch_size, num_channels, data_height, data_width = feature_maps.size()
        num_rois = rois.size(0)

        output = Variable(torch.zeros(num_rois, num_channels,
                                      self.out_h, self.out_w))
        if feature_maps.is_cuda:
            output = output.cuda()

        for roi_idx in range(num_rois):
            batch_idx = int(rois[roi_idx, 0])
            x1, y1, x2, y2 = rois[roi_idx, 1:]
            x1 = x1.data.cpu().numpy()
            y1 = y1.data.cpu().numpy()
            x2 = x2.data.cpu().numpy()
            y2 = y2.data.cpu().numpy()
            
            roi_start_w = round(x1 * self.spatial_scale)
            roi_start_h = round(y1 * self.spatial_scale)
            roi_end_w = round(x2 * self.spatial_scale)
            roi_end_h = round(y2 * self.spatial_scale)
            
            roi_width = max(roi_end_w - roi_start_w, 1)
            roi_height = max(roi_end_h - roi_start_h, 1)
            bin_size_w = float(roi_width) / float(self.out_w)
            bin_size_h = float(roi_height) / float(self.out_h)

            for ph in range(self.out_h):
                for pw in range(self.out_w):
                    hstart = int(np.floor(ph * bin_size_h))
                    wstart = int(np.floor(pw * bin_size_w))
                    hend = int(np.ceil((ph + 1) * bin_size_h))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    
                    hstart = min(data_height, max(0, hstart + roi_start_h))
                    hend = min(data_height, max(0, hend + roi_start_h))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or (wend <= wstart)
                    if is_empty:
                        output[roi_idx, :, ph, pw] = 0
                    else:
                        data = feature_maps[batch_idx]
                        output[roi_idx, :, ph, pw] = torch.max(
                            torch.max(data[:, hstart:hend, wstart:wend], 1, keepdim=True)[0], 2, keepdim=True)[0].view(-1)

        return output

# Define the loss function
class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super(FasterRCNNLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, roi_cls_locs, roi_scores, gt_roi_locs, gt_roi_labels):
        # Compute classification loss
        roi_cls_loss = self.cross_entropy_loss(roi_scores, gt_roi_labels)

        # Compute localization loss
        n_sample = roi_cls_locs.shape[0]
        roi_loc = roi_cls_locs.view(n_sample, -1, 4)
        roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_labels]
        roi_loc_loss = self.smooth_l1_loss(roi_loc, gt_roi_locs)

        return roi_cls_loss + roi_loc_loss

# Define the class rectification function
def class_rectification(rois, roi_scores, equipment_categories, part_categories, aor_threshold=0.8):
    # Divide ROIs into equipment and parts
    equipment_rois = []
    part_rois = []
    for i, roi in enumerate(rois):
        category = torch.argmax(roi_scores[i]).item()
        if category in equipment_categories:
            equipment_rois.append((i, roi))
        elif category in part_categories:
            part_rois.append((i, roi))

    # Sort ROIs by area
    equipment_rois.sort(key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))
    part_rois.sort(key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))

    # Perform class rectification
    for equip_idx, equip_roi in equipment_rois:
        for part_idx, part_roi in part_rois:
            intersection_area = max(0, min(equip_roi[2], part_roi[2]) - max(equip_roi[0], part_roi[0])) * \
                                max(0, min(equip_roi[3], part_roi[3]) - max(equip_roi[1], part_roi[1]))
            part_area = (part_roi[2] - part_roi[0]) * (part_roi[3] - part_roi[1])
            aor = intersection_area / part_area

            if aor >= aor_threshold:
                equip_category = torch.argmax(roi_scores[equip_idx]).item()
                part_category = torch.argmax(roi_scores[part_idx]).item()

                if equipment_categories[equip_category] != part_categories[part_category][:2]:
                    # Rectify part category
                    new_part_category = equipment_categories[equip_category] + "_" + part_categories[part_category][-1]
                    new_part_idx = part_categories.index(new_part_category)
                    roi_scores[part_idx] = 0
                    roi_scores[part_idx][new_part_idx] = 1

    return roi_scores

# Training loop
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

# Main function
def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Initialize model
    model = FasterRCNN(num_classes=args.num_classes)
    model.to(args.device)

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Load data
    train_dataset = InfraredDataset(args.data_path, split='train')
    val_dataset = InfraredDataset(args.data_path, split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    # Training loop
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, args.device, epoch, args.print_freq)
        lr_scheduler.step()
        evaluate(model, val_loader, args.device)

        if args.output_dir:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{epoch}.pth'))

    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on infrared substation images')
    parser.add_argument('--data-path', default='infrared_dataset', help='path to dataset')
    parser.add_argument('--num-classes', default=16, type=int, help='number of object classes (including background)')
    parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr-step-size', default=5, type=int, help='learning rate step size')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='outputs', help='path to save outputs')
    parser.add_argument('--device', default='cuda', help='device (Use cuda or cpu)')
    args = parser.parse_args()

    main(args)
