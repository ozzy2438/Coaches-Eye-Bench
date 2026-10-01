import pandas as pd
import pytest

from ceb.ingest import ingest_stats, ingest_votes
from ceb.join import JoinRateError, check_join_rate, join_votes
from ceb.names import norm_name, parse_vote_name
from ceb.teams import canon_team
from helpers import stats_rows, votes_rows


def test_normalisation_cases():
    assert norm_name("O’Meara") == norm_name("O'Meara") == "omeara"
    assert norm_name("Núñez-Åberg") == "nunez aberg"
    assert norm_name("  De   Goey ") == "de goey"
    p = parse_vote_name("Nick Daicos (Collingwood)")
    assert (p.key, p.init_last, p.club) == ("nickdaicos", "n|daicos", "Collingwood")
    q = parse_vote_name("De Goey, Jordan")
    assert q.key == "jordandegoey" and q.init_last == "j|goey"
    assert parse_vote_name("Jordan De Goey").key == q.key


def test_team_names_and_unknown_raises():
    assert canon_team("Greater Western Sydney") == canon_team("GWS Giants") == "GWS"
    assert canon_team("Footscray") == canon_team("Western Bulldogs")
    assert canon_team("WB") == "Western Bulldogs"
    assert canon_team("ADEL") == "Adelaide"
    assert canon_team("GCFC") == "Gold Coast"
    with pytest.raises(ValueError):
        canon_team("Fitzroy Lions B")


def _join(players, entries, P, **kw):
    st, _ = ingest_stats(stats_rows(players, **kw))
    vo, _ = ingest_votes(votes_rows(entries))
    return join_votes(st, vo, P)


def L(i):  # digits are stripped by name normalisation, so test names use letters only
    return str(i).translate(str.maketrans("0123456789", "abcdefghij"))


def H(i):
    return f"Hfirst{L(i)} Home{L(i)}"


def A(i):
    return f"Afirst{L(i)} Away{L(i)}"


def _full_players():
    return [("Collingwood", f"Hfirst{L(i)}", f"Home{L(i)}", 100 + i) for i in range(22)] + [
        ("Carlton", f"Afirst{L(i)}", f"Away{L(i)}", 200 + i) for i in range(22)]


def _entries_30():
    # 7 receivers summing to 30
    return [(H(0), 9), (H(1), 7), (A(0), 6), (A(1), 4), (H(2), 2), (A(2), 1), (H(3), 1)]


def test_clean_join_includes_match_with_30_votes(P):
    r = _join(_full_players(), _entries_30(), P)
    assert r.report["overall"]["match_rate_rows"] == 1.0
    pm = r.player_match
    assert pm["votes"].sum() == 30 and len(pm) == 44 and r.excluded.empty


def test_initial_tier_and_alias_tier(P):
    pl = [("Collingwood", "Matthew", "Kennedy", 1), ("Carlton", "Zachary", "Smith", 2)] + _full_players()[2:]
    ent = [("Matt Kennedy", 15), ("Zac Smith", 15)]
    r = _join(pl, ent, P)
    assert r.report["overall"]["tiers"] == {"initial": 2}
    # alias: a surname spelled differently is only resolvable through the curated table
    pl2 = [("Collingwood", "Jaeger", "Omeara", 1), ("Carlton", "Zachary", "Smith", 2)] + _full_players()[2:]
    ent2 = [("Jaeger O'Meara-Smith", 15), ("Zachary Smith", 15)]
    st, _ = ingest_stats(stats_rows(pl2))
    vo, _ = ingest_votes(votes_rows(ent2))
    bad = join_votes(st, vo, P)
    assert bad.excluded["reason"].tolist() == ["unmatched_vote_rows"]
    ok = join_votes(st, vo, P, aliases={("jaegeromearasmith", None): "jaegeromeara"})
    assert ok.report["overall"]["tiers"]["alias"] == 1 and ok.excluded.empty


def test_ambiguous_name_resolved_only_with_club_hint(P):
    pl = [("Collingwood", "Josh", "Kennedy", 1), ("Carlton", "Josh", "Kennedy", 2)] + _full_players()[2:]
    no_hint = _join(pl, [("Josh Kennedy", 15), (H(5), 15)], P)
    assert no_hint.unmatched["reason"].tolist() == ["ambiguous_name"]
    assert no_hint.excluded["reason"].tolist() == ["unmatched_vote_rows"]
    hint = _join(pl, [("Josh Kennedy (Carlton)", 15), (H(5), 15)], P)
    assert hint.excluded.empty
    assert hint.player_match.query("player_id == 2")["votes"].iloc[0] == 15
    # AFLCA abbreviates clubs, including the two Bailey Williams' teams in the live pilot.
    clubs = {"Collingwood": "Western Bulldogs", "Carlton": "West Coast"}
    players = [(clubs[t], f, s, i) for t, f, s, i in pl]
    st, _ = ingest_stats(stats_rows(players, home="Western Bulldogs", away="West Coast"))
    vo, _ = ingest_votes(votes_rows([("Josh Kennedy (WB)", 15), (H(5), 15)],
                                   home="Western Bulldogs", away="West Coast"))
    short_hint = join_votes(st, vo, P)
    assert short_hint.excluded.empty
    assert short_hint.player_match.query("player_id == 1")["votes"].iloc[0] == 15


def test_exclusions_for_bad_totals_and_missing_votes(P):
    bad_total = _join(_full_players(), _entries_30()[:-1], P)  # 29 votes
    assert bad_total.excluded["reason"].tolist() == ["vote_total_not_30"]
    assert bad_total.player_match.empty  # never modelled as a zero-vote match
    st, _ = ingest_stats(stats_rows(_full_players()))
    st2, _ = ingest_stats(stats_rows(_full_players(), rnd="2", home="Geelong", away="Essendon",
                                     ).assign(**{"Playing.for": lambda d: d["Playing.for"].map(
                                         {"Collingwood": "Geelong", "Carlton": "Essendon"})}))
    vo, _ = ingest_votes(votes_rows(_entries_30()))
    r = join_votes(pd.concat([st, st2], ignore_index=True), vo, P)
    assert set(r.excluded["reason"]) == {"no_votes_for_match"}
    assert r.player_match["match_id"].nunique() == 1


def test_round_label_mismatch_falls_back_and_duplicate_pages_dropped(P):
    st, _ = ingest_stats(stats_rows(_full_players(), rnd="5"))
    vo_wrong, _ = ingest_votes(votes_rows(_entries_30(), rnd=6))  # AFLCA says 6, box score says 5
    r = join_votes(st, vo_wrong, P)
    assert r.report["round_mismatch_rows"] == 7 and r.excluded.empty
    both = pd.concat([ingest_votes(votes_rows(_entries_30(), rnd=5))[0], vo_wrong], ignore_index=True)
    r2 = join_votes(st, both, P)  # same page served under two round labels
    assert r2.report["duplicate_vote_page_rows_dropped"] == 7 and r2.excluded.empty
    assert r2.player_match["votes"].sum() == 30
    flipped, _ = ingest_votes(votes_rows(_entries_30(), rnd=5, home="Carlton", away="Collingwood"))
    reverse = join_votes(st, flipped, P)  # opposite source orders at a neutral venue
    assert reverse.excluded.empty and reverse.report["home_away_reversal_rows"] == 7
    assert set(reverse.player_match["match_id"]) == set(st["match_id"])
    wrong_round = join_votes(st, flipped.assign(round=6), P)
    assert wrong_round.report["home_away_reversal_rows"] == 0
    assert wrong_round.unmatched["reason"].eq("match_not_in_stats").all()


def test_finals_rows_dropped_at_ingest():
    df = pd.concat([stats_rows(_full_players()), stats_rows(_full_players(), rnd="QF")])
    st, info = ingest_stats(df)
    assert info["non_home_away_rows_dropped"] == 44 and st["round"].eq(1).all()


def test_join_rate_guard_stops_below_floor(P):
    ent = _entries_30()[:-1] + [("Nobody Atall", 1)]
    r = _join(_full_players(), ent, P)
    assert r.report["overall"]["match_rate_rows"] < 1
    with pytest.raises(JoinRateError):
        check_join_rate(r.report, P)


def test_ingest_validates_inputs():
    with pytest.raises(ValueError, match="missing columns"):
        ingest_stats(stats_rows(_full_players()).drop(columns=["Tackles"]))
    with pytest.raises(ValueError, match="missing final scores"):
        ingest_stats(stats_rows(_full_players(), hs=None))
    with pytest.raises(ValueError, match="unknown team"):
        ingest_stats(stats_rows(_full_players()).assign(**{"Playing.for": "Atlantis"}))
