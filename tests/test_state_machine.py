# _max_cyan_ — project_mxsa

import time
from simba.core.state_machine import StateMachine

def test_initial_state():
    sm = StateMachine()
    assert sm.get_state() == "BOOTING"
    assert sm.is_busy() == False
    assert sm.can_accept_command() == False

def test_valid_transitions():
    sm = StateMachine()
    # BOOTING -> IDLE
    assert sm.transition("IDLE") == True
    assert sm.get_state() == "IDLE"
    
    # IDLE -> SCANNING
    assert sm.transition("SCANNING") == True
    assert sm.get_state() == "SCANNING"
    
    # SCANNING -> FETCHING
    assert sm.transition("FETCHING") == True
    assert sm.get_state() == "FETCHING"
    assert sm.is_busy() == True

def test_invalid_transitions():
    sm = StateMachine()
    # BOOTING -> FETCHING is invalid
    assert sm.transition("FETCHING") == False
    assert sm.get_state() == "BOOTING"

def test_context_handling():
    sm = StateMachine()
    sm.transition("IDLE")
    sm.transition("FETCHING", context={"target": "bottle"})
    assert sm.get_state() == "FETCHING"
    assert sm.get_context() == {"target": "bottle"}

def test_state_duration():
    sm = StateMachine()
    time.sleep(0.01)
    duration = sm.get_state_duration()
    assert duration > 0

def test_history():
    sm = StateMachine()
    sm.transition("IDLE")
    sm.transition("SCANNING")
    history = sm.get_history()
    assert len(history) == 2
    assert history[0]["from_state"] == "BOOTING"
    assert history[0]["to_state"] == "IDLE"
    assert history[1]["from_state"] == "IDLE"
    assert history[1]["to_state"] == "SCANNING"
