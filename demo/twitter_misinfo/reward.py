def polarization_reward(state, action, next_state):
    """
    极化最小化奖励：Trump/Biden人数越接近奖励越高。
    """
    beliefs = next_state["beliefs"]
    trump = list(beliefs.values()).count("TRUMP")
    biden = len(beliefs) - trump
    return -abs(trump - biden)

def trump_dominance_reward(state, action, next_state):
    """
    Trump占比最大化奖励。
    """
    beliefs = next_state["beliefs"]
    trump = list(beliefs.values()).count("TRUMP")
    return trump / len(beliefs)

def slot_reward(state, action, next_state):
    """
    Slot reward: 观点slot抢占奖励（如每轮主流观点变化奖励）。
    """
    before = state["beliefs"]
    after = next_state["beliefs"]
    trump_before = list(before.values()).count("TRUMP")
    trump_after = list(after.values()).count("TRUMP")
    # 若主流观点发生变化，奖励+1
    main_before = "TRUMP" if trump_before >= len(before)/2 else "BIDEN"
    main_after = "TRUMP" if trump_after >= len(after)/2 else "BIDEN"
    return 1.0 if main_before != main_after else 0.0 