"""기기 매칭 조건을 재현한 평가와 매칭 파라미터 보정

기기(speakable_match.c)와 같은 점수:
  S_k = α·cos(e, c_k) + (1−α)·max_i cos(e, s_k,i)    c_k: 등록 샘플 평균(정규화 안 함), s_k,i: 등록 샘플
  확실 S1 ≥ T 그리고 S1 − S2 ≥ M / 애매 S1 ≥ T 그리고 S1 − S2 < M (후보 3개) / 거절 S1 < T

에피소드: 화자마다 (화자, 문장) 묶음 중 일부를 "등록 표현"으로 정해 묶음마다 최대 5개를 등록하고,
나머지 발화를 질의로 쓴다. 등록하지 않은 문장의 발화는 "미등록 질의"(거절되어야 함)로 쓴다.
화자 안에서 같은 문장을 반복한 데이터가 부족하면 화자 구분 없이 문장 단위로 평가한다 (mode=text).
"""
from collections import defaultdict
import numpy as np


def make_episodes(speakers, texts, rng, n_enroll=5, unreg_frac=0.2, cross_speaker=False):
    """→ [(enroll: [인덱스 배열...], queries: 인덱스 배열, labels: 묶음 번호 또는 −1)] (화자당 하나)"""
    by_spk = defaultdict(lambda: defaultdict(list))
    for i, (s, t) in enumerate(zip(speakers, texts)):
        by_spk["*" if cross_speaker else s][t].append(i)
    eps = []
    for spk, groups in by_spk.items():
        reps = [np.array(v) for v in groups.values() if len(v) >= 2]
        singles = [np.array(v) for v in groups.values() if len(v) < 2]
        if len(reps) < 2:
            continue
        order = rng.permutation(len(reps))
        n_unreg = int(round(len(reps) * unreg_frac)) if len(reps) >= 4 else 0
        reg = [reps[j] for j in order[n_unreg:]]
        unreg = [reps[j] for j in order[:n_unreg]] + singles
        enroll, q, lab = [], [], []
        for g, idx in enumerate(reg):
            idx = rng.permutation(idx)
            ne = min(n_enroll, len(idx) - 1)
            enroll.append(idx[:ne])
            q += idx[ne:].tolist()
            lab += [g] * (len(idx) - ne)
        for idx in unreg:
            q += idx.tolist()
            lab += [-1] * len(idx)
        eps.append((enroll, np.array(q), np.array(lab)))
    return eps


def score_episode(emb, enroll, queries, alpha):
    """→ 점수 행렬 (질의 수, 등록 표현 수)"""
    e = emb[queries]
    cols = []
    for idx in enroll:
        s = emb[idx]
        proto = s.mean(axis=0)
        c_proto = e @ proto / max(np.linalg.norm(proto), 1e-12)
        cols.append(alpha * c_proto + (1 - alpha) * (e @ s.T).max(axis=1))
    return np.stack(cols, axis=1)


def collect(emb, eps, alpha):
    """에피소드 전체 질의의 S1, S2, 1위 정답 여부, 3위 안 포함 여부, 등록 여부"""
    s1, s2, top1, top3, reg = [], [], [], [], []
    for enroll, q, lab in eps:
        S = score_episode(emb, enroll, q, alpha)
        order = np.argsort(-S, axis=1)
        srt = np.take_along_axis(S, order, axis=1)
        s1.append(srt[:, 0])
        s2.append(srt[:, 1] if S.shape[1] > 1 else np.full(len(q), -1.0))
        top1.append(order[:, 0] == lab)
        top3.append((order[:, :3] == lab[:, None]).any(axis=1))
        reg.append(lab >= 0)
    return {k: np.concatenate(v) for k, v in zip(("s1", "s2", "top1", "top3", "reg"), (s1, s2, top1, top3, reg))}


def decide_rates(c, T, M):
    """등록 질의: 확실·정답 / 확실·오답 / 애매·후보에 정답 / 거절, 미등록 질의: 확실 / 애매 / 거절 비율"""
    r, u = c["reg"], ~c["reg"]
    acc = c["s1"] >= T
    conf = acc & (c["s1"] - c["s2"] >= M)
    amb = acc & ~conf
    n_r, n_u = max(r.sum(), 1), max(u.sum(), 1)
    return {
        "reg_confident_correct": float((conf & c["top1"] & r).sum() / n_r),
        "reg_confident_wrong": float((conf & ~c["top1"] & r).sum() / n_r),
        "reg_ambiguous_hit": float((amb & c["top3"] & r).sum() / n_r),
        "reg_ambiguous_miss": float((amb & ~c["top3"] & r).sum() / n_r),
        "reg_reject": float((~acc & r).sum() / n_r),
        "unreg_confident": float((conf & u).sum() / n_u) if u.any() else None,
        "unreg_ambiguous": float((amb & u).sum() / n_u) if u.any() else None,
        "unreg_reject": float((~acc & u).sum() / n_u) if u.any() else None,
    }


def _search(c, max_wrong, max_unreg, w_amb=0.5):
    best = None
    t_grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    m_grid = np.round(np.arange(0.0, 0.3001, 0.01), 2)
    for T in t_grid:
        for M in m_grid:
            d = decide_rates(c, T, M)
            if d["reg_confident_wrong"] > max_wrong:
                continue
            if d["unreg_confident"] is not None and d["unreg_confident"] > max_unreg:
                continue
            obj = d["reg_confident_correct"] + w_amb * d["reg_ambiguous_hit"]
            if best is None or obj > best[0]:
                best = (obj, float(T), float(M), d)
    return best


def evaluate(emb, speakers, texts, n_trials=3, seed=0, alphas=(0.0, 0.3, 0.5, 0.7, 1.0),
             max_wrong=0.02, max_unreg=0.05, imm_wrong=0.002, imm_unreg=0.005, fixed=None, search=True):
    """emb: L2 정규화 임베딩 (N, 64) numpy. fixed가 있으면 그 파라미터로 평가만 한다 (검증셋으로 정한 값을 시험셋에)
    search=False면 파라미터 탐색 없이 α=0.5의 1위·3위 정답률만 (학습 중 검증용)"""
    emb = np.asarray(emb, dtype=np.float64)
    mode = "speaker"
    eps_all = []
    for t in range(n_trials):
        eps = make_episodes(speakers, texts, np.random.default_rng(seed + t))
        if t == 0 and not eps:
            mode = "text"
        if mode == "text":
            eps = make_episodes(speakers, texts, np.random.default_rng(seed + t), cross_speaker=True)
        eps_all += eps
    if not eps_all:
        return {"mode": mode, "error": "평가할 수 있는 묶음이 없음 (같은 문장 2회 이상인 묶음이 2개 이상 필요)"}

    rep = {"mode": mode, "episodes": len(eps_all), "queries": int(sum(len(q) for _, q, _ in eps_all)),
           "unregistered_queries": int(sum((l < 0).sum() for _, _, l in eps_all))}
    if not search:
        c = collect(emb, eps_all, 0.5)
        rep.update(top1=float(c["top1"][c["reg"]].mean()), top3=float(c["top3"][c["reg"]].mean()))
        return rep
    if fixed is not None:
        c = collect(emb, eps_all, fixed["alpha"])
        rep.update(params=fixed, top1=float(c["top1"][c["reg"]].mean()), top3=float(c["top3"][c["reg"]].mean()),
                   decision=decide_rates(c, fixed["threshold"], fixed["margin"]))
        if fixed.get("immediate_threshold", -1) >= 0:
            rep["immediate"] = decide_rates(c, fixed["immediate_threshold"], fixed["immediate_margin"])
        return rep

    per_alpha = {}
    best = None
    for a in alphas:
        c = collect(emb, eps_all, a)
        r = _search(c, max_wrong, max_unreg)
        per_alpha[a] = {"top1": float(c["top1"][c["reg"]].mean()), "top3": float(c["top3"][c["reg"]].mean()),
                        "objective": None if r is None else r[0]}
        if r is not None and (best is None or r[0] > best[1][0]):
            best = (a, r, c)
    rep["per_alpha"] = per_alpha
    if best is None:
        rep["error"] = f"오답 확실 ≤ {max_wrong}, 미등록 확실 ≤ {max_unreg}를 만족하는 T·M 없음"
        return rep
    a, (_, T, M, d), c = best
    imm = _search(c, imm_wrong, imm_unreg, w_amb=0.0)
    params = {"alpha": a, "threshold": T, "margin": M,
              "immediate_threshold": imm[1] if imm and imm[3]["reg_confident_correct"] >= 0.05 else -1.0,
              "immediate_margin": imm[2] if imm and imm[3]["reg_confident_correct"] >= 0.05 else 0.0}
    params.update(_aux_params(emb, eps_all))
    rep.update(params=params, top1=per_alpha[a]["top1"], top3=per_alpha[a]["top3"], decision=d,
               immediate=imm[3] if params["immediate_threshold"] >= 0 else None,
               constraints={"max_wrong": max_wrong, "max_unreg": max_unreg, "imm_wrong": imm_wrong, "imm_unreg": imm_unreg})
    return rep


def _aux_params(emb, eps):
    """conflict_threshold: 같은 화자의 서로 다른 표현 대표 벡터 cosine의 95백분위
    adapt_min_sim: 등록 질의와 자기 표현 대표 벡터 cosine의 2백분위 (이보다 낮으면 이상치로 봄)"""
    inter, own = [], []
    for enroll, q, lab in eps:
        P = np.stack([emb[i].mean(axis=0) for i in enroll])
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        C = P @ P.T
        inter += C[np.triu_indices(len(P), 1)].tolist()
        r = lab >= 0
        own += np.sum(emb[q[r]] * P[lab[r]], axis=1).tolist()
    return {"conflict_threshold": float(np.percentile(inter, 95)) if inter else 0.9,
            "adapt_min_sim": float(np.percentile(own, 2)) if own else 0.5}


def format_report(rep):
    if "error" in rep and "params" not in rep:
        return f"[{rep.get('mode')}] {rep['error']}"
    lines = [f"모드 {rep['mode']} · 에피소드 {rep['episodes']} · 질의 {rep['queries']} (미등록 {rep['unregistered_queries']})",
             f"등록 질의 1위 정답 {rep['top1']:.3f} · 3위 안 {rep['top3']:.3f}"]
    if "per_alpha" in rep:
        lines.append("α별 1위 정답: " + ", ".join(f"{a}:{v['top1']:.3f}" for a, v in rep["per_alpha"].items()))
    p, d = rep["params"], rep["decision"]
    lines.append(f"파라미터 α={p['alpha']} T={p['threshold']:.2f} M={p['margin']:.2f} "
                 f"즉시 T={p.get('immediate_threshold', -1):.2f} M={p.get('immediate_margin', 0):.2f}")
    lines.append(f"등록: 확실·정답 {d['reg_confident_correct']:.3f}  확실·오답 {d['reg_confident_wrong']:.3f}  "
                 f"애매·후보 적중 {d['reg_ambiguous_hit']:.3f}  애매·후보 실패 {d['reg_ambiguous_miss']:.3f}  거절 {d['reg_reject']:.3f}")
    if d["unreg_confident"] is not None:
        lines.append(f"미등록: 확실(오인식) {d['unreg_confident']:.3f}  애매 {d['unreg_ambiguous']:.3f}  거절 {d['unreg_reject']:.3f}")
    if rep.get("immediate"):
        im = rep["immediate"]
        lines.append(f"즉시 출력: 정답 {im['reg_confident_correct']:.3f}  오답 {im['reg_confident_wrong']:.4f}")
    return "\n".join(lines)
