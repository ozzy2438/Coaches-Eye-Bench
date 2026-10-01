from ceb.config import all_seasons, load_params, load_params_dict, protocol_sha256, smoke_params


def test_protocol_matches_the_brief(P):
    s = P.splits
    assert (s.train_start, s.train_end, list(s.val_seasons), s.test_season) == (2012, 2023, [2024, 2025], 2026)
    assert P.inference.bootstrap_B >= 2000 and P.metrics.k == 5
    assert P.votes.total_per_match == 30 and P.coverage.min_join_rate == 0.99
    assert len(P.features.stats) == 21 and list(P.features.context) == ["won", "margin_10"]
    assert all_seasons(P) == list(range(2012, 2026)) and all_seasons(P, True)[-1] == 2026
    assert P.fantasy.kicks == 3 and P.fantasy.frees_against == -3 and P.fantasy.goals == 6


def test_smoke_params_never_use_the_real_test_season(SP, P):
    assert SP.splits.test_season != P.splits.test_season
    assert SP.inference.bootstrap_B < P.inference.bootstrap_B  # smoke is a plumbing run only


def test_param_block_is_the_single_source_and_hash_is_stable():
    assert load_params_dict()["seed"] == load_params().seed
    assert protocol_sha256() == protocol_sha256() and len(protocol_sha256()) == 64
    assert smoke_params().seed == load_params().seed
