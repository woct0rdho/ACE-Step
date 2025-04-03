from loguru import logger
import functools
import numpy as np
import time
import librosa
import sys
import yaml
from threading import Thread


logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


def async_thread(f):

    def wrapper(*args, **kwargs):
        t = Thread(target=f, args=args, kwargs=kwargs)
        t.start()

    return wrapper


def timecost(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start} seconds to run")
        return result
    return wrapper


def autocut(wav, min_cut_second=9.9, sample_rate=16_000, frame_length=2048, hop_length=512, cut_threshold=[2e-5, 1, 2**0.5], min_mute_duration=120, min_tail_second=2):
    segs = []
    seg_lengths = []
    longest_wav_frames = int(min_cut_second * sample_rate)
    if len(wav) < longest_wav_frames:
        segs.append(wav)
        seg_lengths.append(len(wav))
        return segs, seg_lengths

    # 自适应阈值算法找静音切分点
    candidate_cut_positions = []
    candidate_cut_durations = []
    cut_threshold, cut_threshold_max, cut_step_multiple = cut_threshold

    for i in range(8):
        rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
        is_mute_mask = rms <= cut_threshold
        is_mute = np.zeros_like(rms, dtype='bool')
        is_mute[is_mute_mask], is_mute[~is_mute_mask] = True, False
        # logger.info(f"{rms.mean()=}, {rms.min()=}, {rms.max()=}, {cut_threshold=}, {is_mute_mask.sum()=}")
        last_start = 0
        last_position = 0
        curr_cut_positions = []
        curr_cut_durations = []
        interrupt = False
        for i in range(len(is_mute) - 1):
            # 从有到无
            if not is_mute[i] and is_mute[i + 1]:
                last_start = i
            # 从无到有
            if is_mute[i] and not is_mute[i + 1]:
                # 静音部分至少大于等于min_mute_duration
                mute_duration = (i - last_start) * \
                    hop_length / (sample_rate / 1000)
                if mute_duration >= min_mute_duration:
                    # 切分规则：在静音中间部分作为分割点
                    # 还原到wav的帧
                    mid = (i + last_start) // 2
                    cut_position = mid * hop_length
                    curr_duration = cut_position - last_position
                    # 若超了，切分成四份
                    if (longest_wav_frames // 2) < curr_duration:
                        left_cut_position = (last_start+mid) // 2 * hop_length
                        left_curr_duration = left_cut_position - last_position
                        curr_cut_positions.append(left_cut_position)
                        curr_cut_durations.append(left_curr_duration)
                        last_position = left_cut_position

                        right_cut_position = (mid+i) // 2 * hop_length
                        right_curr_duration = right_cut_position - last_position
                        curr_cut_positions.append(right_cut_position)
                        curr_cut_durations.append(right_curr_duration)
                        last_position = right_cut_position
                    else:
                        curr_cut_positions.append(cut_position)
                        curr_cut_durations.append(curr_duration)
                        last_position = cut_position

        candidate_cut_positions = curr_cut_positions
        candidate_cut_durations = curr_cut_durations
        if cut_threshold >= cut_threshold_max:
            break
        if cut_threshold < cut_threshold_max:
            if len(curr_cut_durations) == 0:
                curr_cut_positions.append(len(wav))
                curr_cut_durations.append(len(wav))
            else:
                curr_cut_positions.append(len(wav))
                curr_cut_durations.append(
                    curr_cut_positions[-1] - curr_cut_positions[-2])
            max_duration = max(curr_cut_durations)
            if max_duration >= longest_wav_frames:
                interrupt = True
                cut_threshold = cut_threshold * cut_step_multiple
                min_mute_duration = int(max(min_mute_duration/cut_step_multiple, 10))
                frame_length = int(max(frame_length / cut_step_multiple, 256))
                hop_length = int(max(hop_length / cut_step_multiple, 64))
                # logger.info(f"Adaptively adjust the threshold: {cut_threshold=} {min_mute_duration=} {frame_length=} {hop_length=} {len(curr_cut_durations)=}")
        if not interrupt and len(curr_cut_durations) > 0:
            candidate_cut_positions = curr_cut_positions
            candidate_cut_durations = curr_cut_durations
            break

    # logger.info(f"candidate_cut_positions {candidate_cut_positions}")
    # logger.info(f"candidate_cut_durations {candidate_cut_durations}")
    # 从已有切分点中找最接近最大长度的切分点
    curr_duration = 0
    last_start = 0
    for i, duration in enumerate(candidate_cut_durations):
        curr_duration += duration
        # 若超出最大限制，以上一个点作为实际切分
        if curr_duration > longest_wav_frames:
            segs.append(wav[last_start:candidate_cut_positions[i - 1]])
            seg_lengths.append(curr_duration - duration)
            curr_duration = duration
            last_start = candidate_cut_positions[i - 1]
    if len(candidate_cut_durations) == 0 or (len(candidate_cut_durations)==1 and candidate_cut_durations[0] >= len(wav)):
        logger.info("自动切分算法失败，按最长强制切分")
        # 按最长强制切分
        last_start = 0
        segs = []
        seg_lengths = []
        for end in range(longest_wav_frames, max(longest_wav_frames, len(wav)), longest_wav_frames):
            segs.append(wav[last_start:end])
            seg_lengths.append(end-last_start)
            last_start = end
    # 解决尾部问题
    if sum(seg_lengths) < len(wav):
        for end in range(last_start+longest_wav_frames, max(longest_wav_frames, len(wav)), longest_wav_frames):
            segs.append(wav[last_start:end])
            seg_lengths.append(end - last_start)
            last_start = end
        if sum(seg_lengths) < len(wav):
            last_start = sum(seg_lengths)
            tail_frame = len(wav) - last_start
            if len(segs) > 0 and tail_frame < min_tail_second*sample_rate:
                segs.pop()
                seg_lengths.pop()
                last_start = sum(seg_lengths)
            segs.append(wav[last_start:])
            seg_lengths.append(len(wav) - last_start)

    if any([len(seg) > longest_wav_frames for seg in segs]):
        new_segs = []
        new_seg_lengths = []
        for seg, seg_length in zip(segs, seg_lengths):
            num_cut = len(seg) // longest_wav_frames
            num_cut += 1 if len(seg) % longest_wav_frames > 0 else 0
            for i in range(num_cut):
                new_segs.append(seg[i*longest_wav_frames:(i+1)*longest_wav_frames])
                new_seg_lengths.append(len(new_segs[-1]))
        segs, seg_lengths = new_segs, new_seg_lengths
    return segs, seg_lengths


class ConfigObj:

    def __init__(self, d):
        self.__dict__.update(d)

    def __repr__(self) -> str:
        return repr(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __getitem__(self, k):
        return self.__dict__[k]

    def get(self, k, default=None):
        if k in self.__dict__:
            return self[k]
        else:
            return default

    def __setitem__(self, k, v):
        self.__dict__[k] = v


def load_config(config_path):
    with open(config_path, encoding='utf-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return ConfigObj(config)
