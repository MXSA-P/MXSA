from simba.core.emotions import EmotionEngine

engine = EmotionEngine({"emotions": {"reactions": {"happy": None}}})

# Test set_emotion with wrong types
engine.set_emotion("happy", "0.9")
engine.set_emotion(None, 0.5)

# Test None reaction config
print("Motor:", engine.get_motor_behavior())

# Test None in emotions config
engine2 = EmotionEngine({"emotions": None})
print("Decay:", engine2.decay_rate)
print("Interval:", engine2.update_interval)

print("All tests passed.")
