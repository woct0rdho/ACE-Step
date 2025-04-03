import torch.nn as nn
import torch
import random
from loguru import logger

from transformers import UMT5EncoderModel, AutoTokenizer, AutoModel
import re
from typing import List, Tuple, Dict, Set
import sys
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_from(self, s: str, start: int) -> List[str]:
        """
        从字符串 s 的位置 start 开始，使用 Trie 树查找所有可能的匹配 phoneme。
        返回所有匹配的 phoneme。
        """
        node = self.root
        matches = []
        current_phoneme = []
        for i in range(start, len(s)):
            char = s[i]
            if char in node.children:
                node = node.children[char]
                current_phoneme.append(char)
                if node.is_end_of_word:
                    matches.append(''.join(current_phoneme))
            else:
                break
        return matches


class PhonemeMatcher:
    def __init__(self, word_dict: Set[str]):
        """
        初始化 PhonemeMatcher，构建 Trie 树。

        :param word_dict: Set[str] - 包含所有 phoneme 的集合
        """
        self.trie = Trie()
        for word in word_dict:
            self.trie.insert(word)

    def tokenize(self, s: str) -> List[str]:
        """
        将输入的 xsampa 字符串拆分成 phoneme 序列，尽可能使用词表中的 phoneme，
        并在无法完全匹配时，选择编辑距离最小且 phoneme 数量最少的序列。

        :param s: str - 输入的 xsampa 字符串
        :return: List[str] - 输出的 phoneme 序列
        """
        n = len(s)
        # 初始化 DP 数组，dp[i] = (cost, phoneme_count, phone_list)
        dp: List[Tuple[int, int, List[str]]] = [(sys.maxsize, sys.maxsize, []) for _ in range(n + 1)]
        dp[0] = (0, 0, [])

        for i in range(n):
            current_cost, current_count, current_list = dp[i]
            if current_cost == sys.maxsize:
                continue  # 无法到达当前位置

            # 查找所有从位置 i 开始的匹配 phoneme
            matches = self.trie.search_from(s, i)

            if matches:
                for phoneme in matches:
                    end = i + len(phoneme)
                    new_cost = current_cost  # 匹配成功，无需增加编辑距离
                    new_count = current_count + 1
                    new_list = current_list + [phoneme]

                    if new_cost < dp[end][0]:
                        dp[end] = (new_cost, new_count, new_list)
                    elif new_cost == dp[end][0]:
                        if new_count < dp[end][1]:
                            dp[end] = (new_cost, new_count, new_list)
            else:
                # 没有匹配的 phoneme，考虑跳过当前字符，增加编辑距离
                new_cost = current_cost + 1
                end = i + 1
                new_count = current_count + 1  # 跳过一个字符也算作一个 phoneme
                new_list = current_list + [s[i]]

                if new_cost < dp[end][0]:
                    dp[end] = (new_cost, new_count, new_list)
                elif new_cost == dp[end][0]:
                    if new_count < dp[end][1]:
                        dp[end] = (new_cost, new_count, new_list)

        # 如果无法完全匹配，选择最优的近似匹配
        if dp[n][0] == sys.maxsize:
            # 找到所有可能的最小编辑距离
            min_cost = min(dp[i][0] for i in range(n + 1))
            # 选择最小编辑距离且 phoneme 数量最少的序列
            candidates = [dp[i] for i in range(n + 1) if dp[i][0] == min_cost]
            if candidates:
                # 选择 phoneme 数量最少的
                best = min(candidates, key=lambda x: x[1])
                return best[2]
            else:
                return []

        return dp[n][2]


HARMONIX_LABELS = [
    'start',
    'end',
    'intro',
    'outro',
    'break',
    'bridge',
    'inst',
    'solo',
    'verse',
    'chorus',
]


def timestamp2second(timestamps):
    res = []
    for item in timestamps:
        start, end = item["start"], item["end"]
        # convert 8kHz to latents level
        start = round(start / 8000, 2)
        end = round(end / 8000, 2)
        res.append({"start": start, "end": end})
    return res


def sample_lyric_mask(voiced_timestamp, max_length):
    voiced_timestamps = timestamp2second(voiced_timestamp)

    min_gaps = [1,2,3,4,5]
    while len(min_gaps) > 0:
        min_gap = min_gaps.pop()
        can_split_breaks = []
        last_end = 0.00
        for item in voiced_timestamps:
            if item["start"] - last_end >= min_gap:
                if last_end == 0.00:
                    can_split_breaks.append((last_end, item["start"] - 0.5))
                else:
                    can_split_breaks.append((last_end + 0.5, item["start"] - 0.5))
            last_end = item["end"]
        if len(can_split_breaks) > 1:
            if can_split_breaks[1][0] <= 360:
                break
            else:
                if min_gap == 1:
                    return 0.0, 360.0, 36

    if len(can_split_breaks) == 0:
        mask_start, mask_end = 0.0, max_length
        min_cut_level = int(mask_end//10 - mask_start//10 + 1)
        return 0.0, mask_end, min_cut_level

    if len(can_split_breaks) == 1:
        # 前后随机选一个
        mask_start = random.choice(["start", "middle"])
        if mask_start == "start":
            mask_start = 0.0
            mask_end = random.uniform(can_split_breaks[0][0], can_split_breaks[0][1])
        else:
            mask_start = random.uniform(can_split_breaks[0][0], can_split_breaks[0][1])
            mask_end = max_length
        min_cut_level = int(mask_end//10 - mask_start//10 + 1)
        return mask_start, mask_end, min_cut_level

    mask_start, mask_end = 0.0, 370
    min_cut_level = 37
    breaths_gap = [end-start for start, end in can_split_breaks]
    max_tried = 5
    while mask_end - mask_start > 370 and min_cut_level > 0 and min_cut_level > 36:
        total_breaths = len(can_split_breaks)
        start = random.choices(range(total_breaths-1), weights=breaths_gap[:-1])[0]
        end = random.choices(range(start + 1, total_breaths), weights=breaths_gap[start+1:], k=1)[0]
        start_break, end_break = can_split_breaks[start], can_split_breaks[end]
        mask_start, mask_end = random.uniform(start_break[0], start_break[1]), random.uniform(end_break[0], end_break[1])
        min_cut_level = int(mask_end//10 - mask_start//10 + 1)
        if min_cut_level < 36:
            min_cut_level = random.randint(min_cut_level, 36)
        if max_tried == 0:
            print("max tried", mask_start, mask_end, min_cut_level, "breaths_gap", breaths_gap, "can_split_breaks", can_split_breaks)
            break
        max_tried -= 1
    mask_start, mask_end = round(mask_start, 2), min(round(mask_end, 2), max_length)
    return mask_start, mask_end, min_cut_level


def check_valid_lyric_lines(lyric_lines):
    # must has lyric lines
    if len(lyric_lines) == 0:
        return False
    for valid_lyric_line in lyric_lines:
        if len(valid_lyric_line[1]) > 0:
            return True
    return False


def select_valid_lyric_lines(lyric_lines, mask_start, mask_end):
    # 选歌词原则
    # 宁可多，不可少
    # 选取mask_start和mask_end之间的歌词行，如果mask_end在一个歌词行中间，那么这个歌词行也要被选取，但最后的structure不要
    valid_lyric_lines = []
    add_tail_structure = True
    for lyric_line in lyric_lines:
        if lyric_line["start"] > lyric_line["end"]:
            continue
        if lyric_line["start"]+1.0 >= mask_start and lyric_line["end"]-1.0 <= mask_end:
            if len(valid_lyric_lines) > 0:
                if valid_lyric_lines[-1][0] is not None and valid_lyric_lines[-1][0] != lyric_line["structure"] and lyric_line["structure"] != "":
                    valid_lyric_lines.append((lyric_line["structure"], [], [], (lyric_line["start"], lyric_line["end"])))
            elif lyric_line["structure"] != "":
                valid_lyric_lines.append((lyric_line["structure"], [], [], (lyric_line["start"], lyric_line["end"])))
            lyric_line["lyric_line"] = lyric_line["lyric_line"].strip()
            if lyric_line["lyric_line"] and "phoneme_line_ipa" in lyric_line and len(lyric_line["phoneme_line_ipa"]) > 0:
                valid_lyric_lines.append((None, lyric_line["lyric_line"], lyric_line["phoneme_line_ipa"], (lyric_line["start"], lyric_line["end"])))
        elif mask_start < lyric_line["start"] and lyric_line["start"] < mask_end and lyric_line["end"] > mask_end:
            lyric_line["lyric_line"] = lyric_line["lyric_line"].strip()
            if lyric_line["lyric_line"] and "phoneme_line_ipa" in lyric_line and len(lyric_line["phoneme_line_ipa"]) > 0:
                valid_lyric_lines.append((None, lyric_line["lyric_line"], lyric_line["phoneme_line_ipa"], (lyric_line["start"], lyric_line["end"])))
                add_tail_structure = False
                break
        elif lyric_line["start"] > mask_start and lyric_line["start"] < mask_end and not lyric_line["lyric_line"] and add_tail_structure:
            valid_lyric_lines.append((lyric_line["structure"], [], [], (lyric_line["start"], lyric_line["end"])))
            add_tail_structure = False
            break
    if len(valid_lyric_lines) > 0 and len(lyric_lines) > 0 and add_tail_structure:
        if lyric_lines[-1]["structure"] != "" and lyric_lines[-1]["structure"] != valid_lyric_lines[-1][0]:
            if lyric_lines[-1]["start"] > mask_start and lyric_lines[-1]["start"] < mask_end:
                valid_lyric_lines.append((lyric_lines[-1]["structure"], [], [], (lyric_lines[-1]["start"], lyric_lines[-1]["end"])))
    return valid_lyric_lines


def sample_lyric_mask_with_cut_levels(voiced_timestamp, cut_level, n_chunks, lyric_lines):
    voiced_timestamps = timestamp2second(voiced_timestamp)

    candidate_spans = []
    for candidate_start_idx in range(n_chunks):
        candidate_start_second = candidate_start_idx * 10
        candidate_end_second = (candidate_start_idx + cut_level) * 10
        valid = True
        for item in voiced_timestamps:
            if item["start"] < candidate_start_second and candidate_start_second < item["end"]:
                valid = False
                break
            if item["start"] < candidate_end_second and candidate_end_second < item["end"]:
                valid = False
                break
        valid_lyric_lines = select_valid_lyric_lines(lyric_lines, candidate_start_second, candidate_end_second)
        if not check_valid_lyric_lines(valid_lyric_lines):
            valid = False
        if valid:
            candidate_spans.append((candidate_start_second, candidate_end_second, valid_lyric_lines))

    if len(candidate_spans) > 0:
        return candidate_spans
    else:
        candidate_spans = []
        for candidate_start_idx in range(n_chunks):
            candidate_start_second = candidate_start_idx * 10
            candidate_end_second = (candidate_start_idx + cut_level) * 10
            valid_lyric_lines = select_valid_lyric_lines(lyric_lines, candidate_start_second, candidate_end_second)
            if check_valid_lyric_lines(valid_lyric_lines):
                candidate_spans.append((candidate_start_second, candidate_end_second, valid_lyric_lines))
        if len(candidate_spans) > 0:
            return candidate_spans
        return []


def sample_lyric_mask_with_lyric_timestamp(cut_level, lyric_lines, expected_num_example, n_chunks, start_pad_offset=1.0):
    # 1 去掉structure
    # non_structure_lyric_lines = [lyric_line for lyric_line in lyric_lines if lyric_line["lyric_line"] and "phoneme_line_ipa" in lyric_line and len(lyric_line["phoneme_line_ipa"]) > 0 and lyric_line["start"] < lyric_line["end"]]
    # 保留structure
    valid_lyric_lines = []
    last_structure = ""
    for lyric_line in lyric_lines:
        if "structure" not in lyric_line:
            lyric_line["structure"] = ""
        if lyric_line["start"] < lyric_line["end"]:
            new_line = lyric_line.copy()
            if not lyric_line["lyric_line"] or "phoneme_line_ipa" not in lyric_line or len(lyric_line["phoneme_line_ipa"]) == 0:
                if lyric_line["structure"] != "":
                    new_line["lyric_line"] = "["+lyric_line["structure"]+"]"
                    new_line["phoneme_line_ipa"] = ["_"]
                else:
                    last_structure = lyric_line["structure"]
                    continue
            else:
                if new_line["structure"] != "" and new_line["structure"] != last_structure:
                    if new_line["lyric_line"] != "[" + new_line["structure"] + "]":
                        new_line["lyric_line"] = f"[{new_line['structure']}]\n{new_line['lyric_line']}"
                        new_line["phoneme_line_ipa"] = ["_", "_"] + new_line["phoneme_line_ipa"]

            valid_lyric_lines.append(new_line)
            last_structure = lyric_line["structure"]

    # 2 优先选刚好包含在里面的
    full_spans = []
    partial_spans = []
    # print("non_structure_lyric_lines", non_structure_lyric_lines, n_chunks)
    for start_idx in range(len(valid_lyric_lines)):
        for end_idx in range(start_idx, len(valid_lyric_lines)):
            start = valid_lyric_lines[start_idx]["start"]
            end = start + cut_level * 10

            # print("start_idx:", start_idx, "end_idx:", end_idx, "start:", start, "end:", end, "non_structure_lyric_lines[end_idx]:", non_structure_lyric_lines[end_idx])

            if start_idx == end_idx and valid_lyric_lines[start_idx]["end"] > end:
                res = [(None, valid_lyric_lines[start_idx]["lyric_line"], valid_lyric_lines[start_idx]["phoneme_line_ipa"], (valid_lyric_lines[start_idx]["start"], valid_lyric_lines[start_idx]["end"])) for line_idx in range(start_idx, end_idx+1)]
                if len(res) > 0:
                    partial_spans.append((start, end, res))
                    break

            if end_idx > 0 and end < valid_lyric_lines[end_idx]["start"] and valid_lyric_lines[end_idx-1]["end"] + start_pad_offset < end:
                res = [(None, valid_lyric_lines[line_idx]["lyric_line"], valid_lyric_lines[line_idx]["phoneme_line_ipa"], (valid_lyric_lines[line_idx]["start"], valid_lyric_lines[line_idx]["end"])) for line_idx in range(start_idx, end_idx)]
                if len(res) > 0:
                    full_spans.append((start, end, res))
                    break

            if end < valid_lyric_lines[end_idx]["end"] + start_pad_offset and end > valid_lyric_lines[end_idx]["start"]:
                res = [(None, valid_lyric_lines[line_idx]["lyric_line"], valid_lyric_lines[line_idx]["phoneme_line_ipa"], (valid_lyric_lines[line_idx]["start"], valid_lyric_lines[line_idx]["end"])) for line_idx in range(start_idx, end_idx)]
                if len(res) > 0:
                    partial_spans.append((start, end, res))
                    break

            if valid_lyric_lines[end_idx]["start"] > end:
                break

        if start_idx == 0 and end_idx == len(valid_lyric_lines) - 1 and len(full_spans) == 0 and len(partial_spans) == 0:
            res = [(None, valid_lyric_lines[line_idx]["lyric_line"], valid_lyric_lines[line_idx]["phoneme_line_ipa"], (valid_lyric_lines[line_idx]["start"], valid_lyric_lines[line_idx]["end"])) for line_idx in range(start_idx, end_idx+1)]
            if len(res) > 0:
                full_spans.append((start, end, res))
    if expected_num_example is not None:
        if len(full_spans) >= expected_num_example or len(partial_spans) == 0:
            return full_spans
        if len(full_spans) + len(partial_spans) >= expected_num_example:
            left = expected_num_example - len(full_spans)
            return full_spans + random.sample(partial_spans, left)
    # print("full_spans:", full_spans)
    # print("partial_spans:", partial_spans)
    return full_spans + partial_spans


class LyricProcessor(nn.Module):
    def __init__(self, infer=False):
        super().__init__()
        self.lyric_text_model = UMT5EncoderModel.from_pretrained("./checkpoints/umt5-base", local_files_only=True).eval().half()
        # not required gradient
        self.lyric_text_model.requires_grad_(False)
        self.lyric_text_tokenizer = AutoTokenizer.from_pretrained("./checkpoints/umt5-base", local_files_only=True)


    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.lyric_text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.lyric_text_model.device != device:
            self.lyric_text_model.to(device)
        with torch.no_grad():
            outputs = self.lyric_text_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    def preprocess(self, valid_lyric_lines):
        lyric_texts = []
        ipa_texts = []
        for valid_line in valid_lyric_lines:
            structure, lyric_line, ipa_line = valid_line["structure"], valid_line["lyric"], valid_line["ipa"]
            if len(structure) > 0:
                lyric_texts.append(structure)
            if len(lyric_line) > 0:
                lyric_texts.append(lyric_line)
            if len(structure) == 0 and len(lyric_line) == 0:
                lyric_texts.append("")
            
            if ipa_line != "_":
                ipa_line = self.split_unk(ipa_line.split(" "))
                ipa_line_str = " ".join(ipa_line)
                # 处理掉G2P的bug
                ipa_line_str = re.sub(r'\bz(?:\s+ə\s+z)+\b', "", ipa_line_str)
                ipa_line_str = re.sub(r'\s+', ' ', ipa_line_str).strip()
                ipa_texts.append(ipa_line_str)
            else:
                ipa_texts.append(ipa_line)
        
        lyric_text = "\n".join(lyric_texts)
        ipa_text = " _ ".join(ipa_texts)
        return lyric_text, ipa_text

