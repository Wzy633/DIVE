#!/usr/bin/env python3
"""可视化预测结果（不画 GT）。

在图片上绘制预测框，并保存可视化结果。

要求：
- 预测框筛选/融合逻辑与 `evaluate_iou_multiple.py` 保持一致（按 label 内融合，label 若无框则取该 label top2）
- 不可视化 GT
- 不同 label 的 bbox 颜色区分开
"""

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_xml_annotation(xml_path: Path) -> Tuple[str, List[str]]:
    """解析 XML 标注文件，提取 filename 和所有 object/name。
    
    Args:
        xml_path: XML 文件路径
        
    Returns:
        (filename, names)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 提取 filename
    filename_elem = root.find('filename')
    filename = filename_elem.text if filename_elem is not None else None

    names: List[str] = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is not None and name_elem.text:
            names.append(name_elem.text)

    return filename, names


def parse_json_prediction(
    json_path: Path, score_threshold: float = 0.0
) -> Tuple[List[Tuple[float, float, float, float, float, int]], Dict[int, str], bool]:
    """解析 JSON 预测文件，提取 score 大于阈值的 bboxes 及其 labels，以及 label_to_class 映射。
    如果没有满足阈值的 bbox，则返回 score 最大的两个 bbox。
    
    Args:
        json_path: JSON 文件路径
        score_threshold: 分数阈值
        
    Returns:
        (边界框列表, label_to_class 映射, 是否使用了后备方案)
        边界框格式为 (xmin, ymin, xmax, ymax, score, label)
        label_to_class 格式为 {label: class_name}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_bboxes = []  # 存储所有 bbox（用于后备方案）
    bboxes = []  # 存储满足阈值的 bbox

    # 读取 label_to_class 映射（由 image_demo_dior_multiple.py 写入）
    label_to_class: Dict[int, str] = {}
    if isinstance(data, dict) and 'label_to_class' in data and isinstance(data['label_to_class'], dict):
        for k, v in data['label_to_class'].items():
            try:
                label_to_class[int(k)] = str(v)
            except Exception:
                # 如果 key 不是可转 int 的，跳过
                continue
    
    # 检查 JSON 格式
    if isinstance(data, dict):
        # mmdetection 格式：{"labels": [...], "scores": [...], "bboxes": [[x1,y1,x2,y2], ...]}
        if 'bboxes' in data and 'scores' in data and 'labels' in data:
            bboxes_data = data['bboxes']
            scores_data = data['scores']
            labels_data = data['labels']
            
            for i, bbox in enumerate(bboxes_data):
                if i < len(scores_data) and i < len(labels_data) and len(bbox) == 4:
                    score = scores_data[i]
                    label = labels_data[i]
                    xmin, ymin, xmax, ymax = bbox
                    bbox_tuple = (xmin, ymin, xmax, ymax, score, int(label))
                    all_bboxes.append(bbox_tuple)
                    
                    if score >= score_threshold:
                        bboxes.append(bbox_tuple)
        # 其他可能的格式
        elif 'predictions' in data:
            predictions = data['predictions']
            for pred in predictions:
                if 'bbox' in pred and 'score' in pred:
                    score = pred['score']
                    label = int(pred.get('label', 0))
                    bbox = pred['bbox']
                    if len(bbox) == 4:
                        xmin, ymin, xmax, ymax = bbox
                        bbox_tuple = (xmin, ymin, xmax, ymax, score, label)
                        all_bboxes.append(bbox_tuple)
                        
                        if score >= score_threshold:
                            bboxes.append(bbox_tuple)
    elif isinstance(data, list):
        # 如果是列表格式
        for item in data:
            if 'bbox' in item and 'score' in item:
                score = item['score']
                label = int(item.get('label', 0))
                bbox = item['bbox']
                if len(bbox) == 4:
                    xmin, ymin, xmax, ymax = bbox
                    bbox_tuple = (xmin, ymin, xmax, ymax, score, label)
                    all_bboxes.append(bbox_tuple)
                    
                    if score >= score_threshold:
                        bboxes.append(bbox_tuple)
    
    # 如果没有满足阈值的 bbox，返回 score 最大的两个
    used_fallback = False
    if len(bboxes) == 0 and len(all_bboxes) > 0:
        # 按 score 排序（降序）
        all_bboxes.sort(key=lambda x: x[4], reverse=True)  # x[4] 是 score
        # 取前两个
        bboxes = all_bboxes[:2]
        used_fallback = True
    
    return bboxes, label_to_class, used_fallback


def calculate_iou(bbox1: Tuple[float, float, float, float], 
                  bbox2: Tuple[float, float, float, float]) -> float:
    """计算两个边界框的 IoU。
    
    Args:
        bbox1: (x1, y1, x2, y2) 格式的边界框
        bbox2: (x1, y1, x2, y2) 格式的边界框
        
    Returns:
        IoU 值
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def suppress_overlapping_bboxes_by_iou_for_label(
    pred_bboxes: List[Tuple[float, float, float, float, float]],
    iou_threshold: float = 0.5,
    keep_score_threshold: float = 0.4,
) -> List[Tuple[float, float, float, float, float]]:
    """对单个 label 的 bbox 进行融合（简易 NMS）：
    - score >= keep_score_threshold 的 bbox 全部保留，不参与合并
    - 仅对 score < keep_score_threshold 的 bbox 计算 IoU 并合并：若 IoU > iou_threshold，只保留 score 更大的

    低分部分采用按 score 降序的贪心策略：依次取最高分 bbox，移除与其 IoU 超过阈值的其它 bbox。

    Args:
        pred_bboxes: [(xmin, ymin, xmax, ymax, score), ...]
        iou_threshold: 抑制阈值，默认 0.5
        keep_score_threshold: 高分保留阈值，默认 0.4（>= 0.4 全部保留）

    Returns:
        过滤后的 bbox 列表（仍包含 score）
    """
    if not pred_bboxes:
        return pred_bboxes

    # 1) 高分 bbox 全部保留
    high_score = [b for b in pred_bboxes if b[4] >= keep_score_threshold]

    # 2) 仅对低分 bbox 做合并
    low_score = [b for b in pred_bboxes if b[4] < keep_score_threshold]

    # 按 score 从高到低排序（仅低分部分）
    sorted_bboxes = sorted(low_score, key=lambda x: x[4], reverse=True)
    kept: List[Tuple[float, float, float, float, float]] = []

    for cand in sorted_bboxes:
        cand_coords = cand[:4]
        should_keep = True
        for kept_box in kept:
            kept_coords = kept_box[:4]
            if calculate_iou(cand_coords, kept_coords) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)

    # 最终 bbox：高分全保留 + 低分去重后的 kept
    # 为了稳定输出，整体按 score 降序排序
    merged = high_score + kept
    merged = sorted(merged, key=lambda x: x[4], reverse=True)
    return merged


def suppress_overlapping_bboxes_by_iou_with_fallback(
    all_pred_bboxes: List[Tuple[float, float, float, float, float, int]],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    keep_score_threshold: float = 0.4,
) -> List[Tuple[float, float, float, float, float, int]]:
    """与 `evaluate_iou_multiple.py` 一致的 bbox 选择/融合策略（按 label 内处理，带 fallback）。

    - 对每个 label：
      - 若该 label 存在 score >= score_threshold 的 bbox：对这些 bbox 做融合（label 内）
      - 否则：取该 label 下 score 最大的 2 个 bbox
    """
    if not all_pred_bboxes:
        return []

    bboxes_by_label: Dict[int, List[Tuple[float, float, float, float, float, int]]] = defaultdict(list)
    for b in all_pred_bboxes:
        bboxes_by_label[b[5]].append(b)

    merged_all: List[Tuple[float, float, float, float, float, int]] = []
    for label, label_bboxes in bboxes_by_label.items():
        above = [b for b in label_bboxes if b[4] >= score_threshold]
        if above:
            coords = [b[:5] for b in above]
            merged = suppress_overlapping_bboxes_by_iou_for_label(
                coords, iou_threshold=iou_threshold, keep_score_threshold=keep_score_threshold
            )
            merged_all.extend([(*m, label) for m in merged])
        else:
            top2 = sorted(label_bboxes, key=lambda x: x[4], reverse=True)[:2]
            merged_all.extend(top2)

    merged_all.sort(key=lambda x: x[4], reverse=True)
    return merged_all


def draw_bbox(img: np.ndarray, bbox: Tuple[float, float, float, float], 
              color: Tuple[int, int, int], thickness: int = 3, label: str = None):
    """在图片上绘制边界框。
    
    Args:
        img: 图片数组 (BGR 格式)
        bbox: 边界框 (xmin, ymin, xmax, ymax)
        color: 颜色 (B, G, R)
        thickness: 线条粗细
        label: 可选的标签文本
    """
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    
    # 获取图片尺寸
    img_height, img_width = img.shape[:2]
    
    # 绘制矩形
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    
    # 如果有标签，绘制文本（调整位置避免超出边界）
    if label:
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # 计算文本背景的位置
        text_bg_height = text_height + baseline + 5
        text_bg_width = text_width
        
        # 尝试在左上角（bbox 外部）
        text_bg_x1 = xmin
        text_bg_y1 = ymin - text_bg_height
        text_bg_x2 = xmin + text_bg_width
        text_bg_y2 = ymin
        
        # 如果左上角超出边界，尝试放在 bbox 内部左上角
        if text_bg_y1 < 0:
            text_bg_y1 = ymin
            text_bg_y2 = ymin + text_bg_height
            text_y = ymin + text_height + baseline - 2
        else:
            text_y = ymin - baseline - 2
        
        # 如果右侧超出边界，调整到左侧
        if text_bg_x2 > img_width:
            text_bg_x1 = xmax - text_bg_width
            text_bg_x2 = xmax
            text_x = text_bg_x1
        else:
            text_x = xmin
        
        # 如果左侧也超出边界，放在 bbox 内部
        if text_bg_x1 < 0:
            text_bg_x1 = xmin
            text_bg_x2 = xmin + text_bg_width
            text_x = xmin
        
        # 确保文本背景在图片范围内
        text_bg_x1 = max(0, min(text_bg_x1, img_width - 1))
        text_bg_y1 = max(0, min(text_bg_y1, img_height - 1))
        text_bg_x2 = max(0, min(text_bg_x2, img_width - 1))
        text_bg_y2 = max(0, min(text_bg_y2, img_height - 1))
        
        # 确保文本位置在图片范围内
        text_x = max(0, min(text_x, img_width - 1))
        text_y = max(text_height, min(text_y, img_height - 1))
        
        # 绘制文本背景（与 bbox 边框颜色一致）
        text_bg_color = color  # 使用与 bbox 边框相同的颜色
        cv2.rectangle(img, (text_bg_x1, text_bg_y1), 
                     (text_bg_x2, text_bg_y2), text_bg_color, -1)
        
        # 绘制文本（白色）
        cv2.putText(img, label, (text_x, text_y), 
                   font, font_scale, (255, 255, 255), text_thickness)


def _color_for_label(label: int) -> Tuple[int, int, int]:
    """给不同 label 生成稳定、鲜艳的 BGR 颜色。"""
    # HSV -> BGR，保证颜色分散
    hue = (label * 37) % 180  # OpenCV H: [0,179]
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def visualize_predictions_only(
    preds_dir: str,
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    score_threshold: float = 0.25,
):
    """仅可视化预测结果（不画 GT）。
    
    Args:
        preds_dir: 预测结果 JSON 文件目录
        annotations_dir: XML 标注文件目录
        images_dir: 图片文件目录
        output_dir: 输出目录
        score_threshold: 分数阈值
    """
    preds_path = Path(preds_dir)
    annotations_path = Path(annotations_dir)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 JSON 和 XML 文件
    json_files = sorted(preds_path.glob('*.json'))
    xml_files = sorted(annotations_path.glob('*.xml'))
    
    # 创建文件名到文件的映射（不带扩展名）
    json_map = {f.stem: f for f in json_files}
    xml_map = {f.stem: f for f in xml_files}
    
    # 找到匹配的文件对
    matched_pairs = []
    for name in json_map.keys():
        if name in xml_map:
            matched_pairs.append((name, json_map[name], xml_map[name]))
    
    print(f'Found {len(matched_pairs)} matched file pairs')
    
    line_thickness = 4  # 边框粗细
    
    # 处理每个文件对
    for name, json_file, xml_file in matched_pairs:
        try:
            # 从 XML 拿到图片 filename
            filename, gt_names = parse_xml_annotation(xml_file)
            if not filename:
                print(f'Warning: No filename found in {xml_file.name}, skipping')
                continue
            gt_name_set = {n.strip().lower() for n in gt_names if isinstance(n, str) and n.strip()}

            # 读取所有预测 bbox（用于按 label fallback）
            all_pred_bboxes, label_to_class, _ = parse_json_prediction(json_file, score_threshold=0.0)
            
            if not all_pred_bboxes:
                print(f'Warning: No predictions in {json_file.name}, skipping')
                continue

            # 应用 bbox 融合（与 evaluate_iou_multiple.py 中一致的方法）
            pred_bboxes = suppress_overlapping_bboxes_by_iou_with_fallback(
                all_pred_bboxes,
                score_threshold=score_threshold,
                iou_threshold=0.5,
                keep_score_threshold=0.4,
            )
            # 按 score 降序，保证 ID 稳定
            pred_bboxes = sorted(pred_bboxes, key=lambda x: x[4], reverse=True)
            
            # 构建图片路径
            image_path = images_path / filename
            if not image_path.exists():
                print(f'Warning: Image not found: {image_path}, skipping')
                continue
            
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                print(f'Warning: Failed to load image: {image_path}, skipping')
                continue
            
            # 绘制预测框并收集信息：
            # - 左上角标注目标 ID（1,2,3,...）
            # - 若预测类别与 XML 中 name 匹配 -> 红色框
            # - 否则按 label 生成区分颜色
            marker_info_list = []  # 保存每个 ID 对应的 bbox 和类别信息
            for i, pred_bbox in enumerate(pred_bboxes):
                xmin, ymin, xmax, ymax, score, lbl = pred_bbox
                cls = label_to_class.get(int(lbl), None)
                cls_norm = cls.strip().lower() if isinstance(cls, str) else None
                if cls_norm is not None and cls_norm in gt_name_set:
                    color = (0, 0, 255)  # 纯红色(BGR)：类别与 XML name 匹配
                else:
                    color = (0, 255, 0)  # 纯绿色(BGR)：类别与 XML name 不匹配

                # 只显示 ID
                marker_id = i + 1
                text = str(marker_id)
                draw_bbox(img, (xmin, ymin, xmax, ymax), color, line_thickness, text)
                
                # 保存每个 ID 对应的信息
                marker_info = {
                    'marker_id': marker_id,
                    'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)],
                    'class': cls if cls else f"class_{lbl}",
                    'label': int(lbl),
                    'score': float(score)
                }
                marker_info_list.append(marker_info)
            
            # 保存可视化结果
            output_filename = f'{name}.jpg'
            output_path_file = output_path / output_filename
            cv2.imwrite(str(output_path_file), img)
            
            # 保存 ID 对应的 bbox 和类别信息到 JSON 文件
            json_output_filename = f'{name}.json'
            json_output_path_file = output_path / json_output_filename
            with open(json_output_path_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'image_name': name,
                    'markers': marker_info_list
                }, f, indent=2, ensure_ascii=False)
            
            print(f'Processed {name}: Pred={len(pred_bboxes)}, Saved to {json_output_filename}')
            
        except Exception as e:
            print(f'Error processing {name}: {e}')
            continue
    
    print(f'\nVisualization complete! Results saved to {output_dir}')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize predictions (no GT)')
    parser.add_argument('--preds-dir', type=str,
                       required=True,
                       help='Directory containing prediction JSON files')
    parser.add_argument('--annotations-dir', type=str,
                       required=True,
                       help='Directory containing annotation XML files')
    parser.add_argument('--images-dir', type=str,
                       required=True,
                       help='Directory containing image files')
    parser.add_argument('--output-dir', type=str,
                       required=True,
                       help='Output directory for visualized images')
    parser.add_argument('--score-threshold', type=float, default=0.25,
                       help='Score threshold for predictions')
    
    args = parser.parse_args()
    
    print(f'Visualizing predictions only (no GT)...')
    print(f'Predictions directory: {args.preds_dir}')
    print(f'Annotations directory: {args.annotations_dir}')
    print(f'Images directory: {args.images_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Score threshold: {args.score_threshold}')
    print('-' * 60)
    
    visualize_predictions_only(
        args.preds_dir,
        args.annotations_dir,
        args.images_dir,
        args.output_dir,
        args.score_threshold
    )


if __name__ == '__main__':
    main()
