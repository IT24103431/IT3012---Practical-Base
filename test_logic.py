from logic_engine import KnowledgeBase
from agent import SearchAgent


def test_forward_chaining():
    kb = KnowledgeBase()
    kb.tell_rule(["TargetVisible", "HasDust"], "SafeToEngage")
    kb.tell_rule(["SafeToEngage", "BloodseekerMissing"], "Retreat")

    kb.clear_facts()
    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.forward_chain()
    assert "SafeToEngage" in kb.facts
    assert "Retreat" not in kb.facts

    kb.clear_facts()
    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.tell_fact("BloodseekerMissing")
    kb.forward_chain()
    assert "Retreat" in kb.facts

    # A reversed rule order proves that another pass can be required.
    kb.clear_facts()
    kb.rules.reverse()
    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.tell_fact("BloodseekerMissing")
    kb.forward_chain()
    assert "Retreat" in kb.facts


def test_astar_avoids_infeasible_tile():
    agent = SearchAgent()
    tile_facts = {
        (1, 0): ["TargetVisible", "HasDust", "BloodseekerMissing"],
        (0, 1): ["TargetVisible", "HasDust"],
    }
    path = agent.astar_search((0, 0), (2, 0), [], (3, 2),
                              tile_facts=tile_facts)
    assert path is not None
    position = (0, 0)
    deltas = dict(agent._MOVES)
    for action in path:
        dx, dy = deltas[action]
        position = (position[0] + dx, position[1] + dy)
        assert position != (1, 0), "A* entered the infeasible tile"
    assert position == (2, 0)

    percept = {
        "agent_pos": [0, 0],
        "all_food": [[2, 0]],
        "remaining_food": 1,
        "walls": [],
        "grid_size": (3, 2),
    }
    assert agent.sense_and_act(percept) == "Up"


if __name__ == "__main__":
    test_forward_chaining()
    test_astar_avoids_infeasible_tile()
    print("All Logic Engine Test Cases Passed!")
