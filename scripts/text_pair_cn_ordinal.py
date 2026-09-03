"""中文排序位次：第一个、第二个、……（用于 STVG 风格题干，避免与阿拉伯数字 ID 混淆）。"""


def cn_ordinal_rank(n: int) -> str:
    """返回「第…个」形式的中文序数短语，如 1->第一个，11->第十一个。"""
    if n < 1:
        raise ValueError(f"rank must be >= 1, got {n}")
    digits = "一二三四五六七八九"
    if n < 10:
        return f"第{digits[n - 1]}个"
    if 10 <= n <= 19:
        if n == 10:
            return "第十个"
        return f"第十{digits[n - 11]}个"
    tens, ones = divmod(n, 10)
    tens_cn = ["", "十", "二十", "三十", "四十", "五十", "六十", "七十", "八十", "九十"]
    head = tens_cn[tens]
    if ones == 0:
        return f"第{head}个"
    return f"第{head}{digits[ones - 1]}个"


# 排序用语 vs 红字 ID：放在推理 system prompt 中一次即可，勿逐条塞进 question。
RANK_VS_ID_RULE = (
    "说明：「第一个」「第二个」等仅表示同类目标按几何顺序数下来的位次；"
    "请输出该目标旁红字标注的「目标ID」数字，勿把位次序号与红字内的数字混为一谈。"
)
# 旧版曾把同一段话包在括号里附在题干末尾，清洗 JSONL 时按此字面量剥离
RANK_ID_DISAMBIG_LEGACY = f"（{RANK_VS_ID_RULE}）"

# 面积题：与评估脚本一致，按框的像素面积；红字数字不参与面积
AREA_BBOX_RULE = (
    "涉及「面积最大/最小」时，只比较同类检测框矩形的像素面积(框宽×框高)；"
    "红字 ID 仅为标注数字，不计入框面积，也不要用数字的视觉大小代替框面积。"
)
