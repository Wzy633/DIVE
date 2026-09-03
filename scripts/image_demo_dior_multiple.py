# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new infenence class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        python demo/image_demo_dior_multiple.py \
        --inputs /path/to/DIOR-RSVG/TestAnnotations \
        --model configs/lae_dino/lae_dino_swin-t_finetune_DIOR.py \
        --weights /path/to/checkpoint.pth \
        --palette random --out-dir outputs_dior_multiple --batch-size 2 \
        --pred-score-thr 0.4
"""

import ast
import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


def extract_all_class_names(xml_annotations_dir):
    """从所有 XML 文件中提取所有唯一的 name 类别。
    
    Args:
        xml_annotations_dir: XML 标注文件目录
        
    Returns:
        set: 所有唯一的 name 类别集合
    """
    class_names = set()
    xml_files = Path(xml_annotations_dir).glob('*.xml')
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None and name_elem.text:
                    class_names.add(name_elem.text)
        except Exception as e:
            print_log(f'Error parsing {xml_file}: {e}', level='WARNING')
            continue
    
    return class_names


def find_names_in_description(description, class_names):
    """在 description 中查找是否包含 class_names 中的任何类别（不考虑空格，大小写不敏感）。
    
    Args:
        description: description 文本
        class_names: 类别名称集合
        
    Returns:
        list: 在 description 中找到的类别名称列表
    """
    if not description:
        return []
    
    # 将 description 转换为小写并移除空格，用于匹配
    description_normalized = description.lower().replace(' ', '')
    found_names = []
    
    for class_name in class_names:
        # 将类别名称转换为小写并移除空格
        class_name_normalized = class_name.lower().replace(' ', '')
        # 检查 description 中是否包含该类别的名称
        if class_name_normalized in description_normalized:
            found_names.append(class_name)
    
    return found_names


def parse_xml_annotation(xml_path, class_names=None):
    """解析 XML 标注文件，提取 filename、所有 name 标签的内容，以及 description 中匹配的类别。
    
    Args:
        xml_path: XML 文件路径
        class_names: 所有类别名称集合（用于在 description 中查找）
        
    Returns:
        tuple: (filename, names_list) 其中 names_list 包括：
            - 所有 <name> 标签的内容
            - description 中匹配到的类别名称
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 提取 filename
    filename_elem = root.find('filename')
    filename = filename_elem.text if filename_elem is not None else None
    
    # 提取所有 <name> 标签的内容
    names = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is not None and name_elem.text:
            names.append(name_elem.text)
        
        # 如果提供了 class_names，从 description 中查找匹配的类别
        if class_names is not None:
            description_elem = obj.find('description')
            if description_elem is not None and description_elem.text:
                description = description_elem.text
                found_names = find_names_in_description(description, class_names)
                # 将找到的类别添加到 names 中（去重）
                for found_name in found_names:
                    if found_name not in names:
                        names.append(found_name)
    
    return filename, names


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs', type=str, required=True,
        help='Input image file or folder path.')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    
    # 检查输入是否是 XML 标注文件夹
    inputs_path = call_args.get('inputs', '')
    xml_annotations_dir = inputs_path
    jpeg_images_dir = os.path.join(os.path.dirname(inputs_path), 'JPEGImages')
    
    # 如果输入是 XML 标注文件夹，处理 XML 文件
    # 支持绝对路径或相对路径，只要路径包含 Annotations 或者是包含 XML 文件的目录
    is_xml_dir = False
    if inputs_path:
        inputs_path_abs = os.path.abspath(inputs_path)
        # 检查是否是已知的标注目录，或者目录中包含 XML 文件
        if (os.path.isdir(inputs_path) and 
            ('Annotations' in inputs_path or 
             any(f.endswith('.xml') for f in os.listdir(inputs_path) if os.path.isfile(os.path.join(inputs_path, f))))):
            is_xml_dir = True
            # 使用用户提供的路径
            xml_annotations_dir = inputs_path
            # 尝试自动推断 JPEGImages 目录
            parent_dir = os.path.dirname(xml_annotations_dir)
            potential_jpeg_dir = os.path.join(parent_dir, 'JPEGImages')
            if os.path.exists(potential_jpeg_dir):
                jpeg_images_dir = potential_jpeg_dir
    
    if is_xml_dir:
        if not os.path.exists(xml_annotations_dir):
            raise ValueError(f"XML annotations directory not found: {xml_annotations_dir}")
        if not os.path.exists(jpeg_images_dir):
            raise ValueError(f"JPEG images directory not found: {jpeg_images_dir}")
        
        # 首先提取所有类别名称（20 类）
        print_log('Extracting all class names from XML annotations...')
        class_names = extract_all_class_names(xml_annotations_dir)
        print_log(f'Found {len(class_names)} unique class names: {sorted(class_names)}')
        
        # 初始化 inferencer
        inferencer = DetInferencer(**init_args)
        chunked_size = call_args.pop('chunked_size', -1)
        inferencer.model.test_cfg.chunked_size = chunked_size
        
        # 获取所有 XML 文件
        xml_files = sorted(Path(xml_annotations_dir).glob('*.xml'))
        print_log(f'Found {len(xml_files)} XML files to process')
        
        # 处理每个 XML 文件
        for xml_file in xml_files:
            try:
                # 解析 XML 文件，传入 class_names 以便从 description 中提取匹配的类别
                filename, names = parse_xml_annotation(xml_file, class_names=class_names)
                
                if not filename:
                    print_log(f'Warning: No filename found in {xml_file}, skipping')
                    continue
                
                if not names:
                    print_log(f'Warning: No names found in {xml_file}, skipping')
                    continue
                
                # 构建图片路径
                image_path = os.path.join(jpeg_images_dir, filename)
                if not os.path.exists(image_path):
                    print_log(f'Warning: Image not found: {image_path}, skipping')
                    continue
                
                # 格式化 texts：将所有 name 用 " . " 连接
                texts = ' . '.join(names) + ' .'
                
                # 获取 XML 文件名（不带扩展名）作为输出文件名
                xml_basename = xml_file.stem
                
                # 设置输出目录为原始输出目录，但使用 XML 文件名
                original_out_dir = call_args.get('out_dir', 'outputs')
                call_args['out_dir'] = original_out_dir
                
                # 更新 call_args
                call_args['inputs'] = image_path
                call_args['texts'] = texts
                call_args['custom_entities'] = True  # 自动启用 -c 选项
                
                print_log(f'Processing {xml_file.name}: image={filename}, texts={texts}')
                
                # 获取图片文件名（不带扩展名），用于后续重命名
                image_basename = Path(filename).stem
                
                # 运行推理
                inferencer(**call_args)
                
                # 保存类别名称映射（label -> class_name）
                # names 列表的顺序对应 label 的索引（0, 1, 2, ...）
                label_to_class = {i: name for i, name in enumerate(names)}
                
                # 立即重命名输出文件，避免多个 XML 指向同一图片时文件被覆盖
                # 添加重试机制，处理批处理导致的文件延迟写入
                if call_args.get('out_dir', '') and not (call_args.get('no_save_vis', False) and call_args.get('no_save_pred', False)):
                    out_dir = call_args['out_dir']
                    if os.path.exists(out_dir):
                        import time
                        max_retries = 10
                        retry_delay = 0.2  # 200ms
                        
                        for retry in range(max_retries):
                            renamed_count = 0
                            
                            # DetInferencer 将结果保存在 preds/ 和 vis/ 子目录中
                            subdirs = ['preds', 'vis']
                            for subdir in subdirs:
                                subdir_path = Path(out_dir) / subdir
                                if subdir_path.exists() and subdir_path.is_dir():
                                    # 在子目录中查找文件
                                    for file in subdir_path.iterdir():
                                        if file.is_file():
                                            # 检查文件名是否匹配图片名（精确匹配）
                                            if file.stem == image_basename:
                                                # 构建新文件名：使用 XML 文件名
                                                new_name = file.parent / f'{xml_basename}{file.suffix}'
                                                
                                                if file != new_name:
                                                    # 如果目标文件已存在，先删除（说明之前已经处理过这个 XML）
                                                    if new_name.exists():
                                                        new_name.unlink()
                                                    file.rename(new_name)
                                                    renamed_count += 1
                                                    
                                                    # 如果是 JSON 文件，添加类别名称映射信息
                                                    if new_name.suffix == '.json':
                                                        try:
                                                            import json
                                                            with open(new_name, 'r') as f:
                                                                data = json.load(f)
                                                            
                                                            # 添加类别名称映射信息
                                                            data['class_names'] = names  # 类别名称列表，索引对应 label
                                                            data['label_to_class'] = label_to_class  # label -> class_name 映射
                                                            
                                                            with open(new_name, 'w') as f:
                                                                json.dump(data, f, indent=2)
                                                        except Exception as e:
                                                            print_log(f'Warning: Failed to add class names to {new_name}: {e}', level='WARNING')
                            
                            # 也检查输出目录根目录中的文件
                            for file in Path(out_dir).iterdir():
                                if file.is_file() and file.stem == image_basename:
                                    new_name = file.parent / f'{xml_basename}{file.suffix}'
                                    if file != new_name:
                                        if new_name.exists():
                                            new_name.unlink()
                                        file.rename(new_name)
                                        renamed_count += 1
                                        
                                        # 如果是 JSON 文件，添加类别名称映射信息
                                        if new_name.suffix == '.json':
                                            try:
                                                import json
                                                with open(new_name, 'r') as f:
                                                    data = json.load(f)
                                                
                                                # 添加类别名称映射信息
                                                data['class_names'] = names  # 类别名称列表，索引对应 label
                                                data['label_to_class'] = label_to_class  # label -> class_name 映射
                                                
                                                with open(new_name, 'w') as f:
                                                    json.dump(data, f, indent=2)
                                            except Exception as e:
                                                print_log(f'Warning: Failed to add class names to {new_name}: {e}', level='WARNING')
                            
                            # 如果找到了文件并重命名成功，跳出重试循环
                            if renamed_count > 0:
                                if retry > 0:
                                    print_log(f'Renamed {renamed_count} file(s) for {xml_file.name} (after {retry+1} attempts)')
                                else:
                                    print_log(f'Renamed {renamed_count} file(s) for {xml_file.name}')
                                break
                            
                            # 如果还没找到文件，等待后重试
                            if retry < max_retries - 1:
                                time.sleep(retry_delay)
                        
                        # 检查是否已经存在以 xml_basename 命名的 JSON 文件（可能之前已经重命名过）
                        # 如果存在，也需要更新类别信息
                        import json
                        for subdir in ['preds', 'vis']:
                            subdir_path = Path(out_dir) / subdir
                            if subdir_path.exists() and subdir_path.is_dir():
                                json_file = subdir_path / f'{xml_basename}.json'
                                if json_file.exists() and json_file.is_file():
                                    try:
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                        
                                        # 添加类别名称映射信息（如果还没有）
                                        if 'class_names' not in data or 'label_to_class' not in data:
                                            data['class_names'] = names
                                            data['label_to_class'] = label_to_class
                                            
                                            with open(json_file, 'w') as f:
                                                json.dump(data, f, indent=2)
                                            print_log(f'Updated class names in {json_file.name}')
                                    except Exception as e:
                                        print_log(f'Warning: Failed to update class names in {json_file}: {e}', level='WARNING')
                        
                        # 也检查根目录
                        json_file = Path(out_dir) / f'{xml_basename}.json'
                        if json_file.exists() and json_file.is_file():
                            try:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                
                                # 添加类别名称映射信息（如果还没有）
                                if 'class_names' not in data or 'label_to_class' not in data:
                                    data['class_names'] = names
                                    data['label_to_class'] = label_to_class
                                    
                                    with open(json_file, 'w') as f:
                                        json.dump(data, f, indent=2)
                                    print_log(f'Updated class names in {json_file.name}')
                            except Exception as e:
                                print_log(f'Warning: Failed to update class names in {json_file}: {e}', level='WARNING')
                        
                        # 如果重试后仍然没有找到文件，输出警告
                        if renamed_count == 0:
                            print_log(f'Warning: No output files found to rename for {xml_file.name} (image={image_basename})')
                
            except Exception as e:
                print_log(f'Error processing {xml_file}: {e}', level='ERROR')
                continue
        
        print_log(f'Finished processing {len(xml_files)} XML files')
        if call_args.get('out_dir', '') and not (call_args.get('no_save_vis', False) and call_args.get('no_save_pred', False)):
            print_log(f'Results saved at {call_args["out_dir"]}')
    
    else:
        # 原始处理逻辑：处理单个图片或图片文件夹
        inferencer = DetInferencer(**init_args)
        
        chunked_size = call_args.pop('chunked_size', -1)
        inferencer.model.test_cfg.chunked_size = chunked_size
        
        inferencer(**call_args)
        
        if call_args.get('out_dir', '') and not (call_args.get('no_save_vis', False) and call_args.get('no_save_pred', False)):
            print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
