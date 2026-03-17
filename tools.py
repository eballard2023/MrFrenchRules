# """
# Profile extraction tools for the Train Jamie interview.
# Maintains expert profile (VAD, OCEAN, attachment, directive, confidence) in DB cache keyed by session_id.
# """

# import json
# import logging
# from langchain_core.tools import tool

# logger = logging.getLogger(__name__)

# # -----------------------------
# # In-memory profile storage (keyed by session_id / user_id)
# # -----------------------------
# DB = {}


# @tool
# def update_vad(session_id: str, vad: dict) -> str:
#     """
#     Update the Valence-Arousal-Dominance metrics for the given session.

#     Parameters:
#     - session_id (str): Unique identifier for the conversation session (use the session_id from context).
#     - vad (dict): Dictionary with keys 'valence', 'arousal', 'dominance', float values 0.0-1.0.

#     Example:
#     update_vad("session_1", {"valence": 0.75, "arousal": 0.32, "dominance": 0.6})
#     """
#     print("update_vad", DB)
#     if session_id not in DB:
#         DB[session_id] = {}
#     DB[session_id]["vad"] = vad
#     logger.debug("Profile updated: vad for %s", session_id)
#     return json.dumps({"ok": True, "vad": vad})


# @tool
# def update_ocean(session_id: str, ocean: dict) -> str:
#     """
#     Update the OCEAN (Big Five) personality metrics for the given session.

#     Parameters:
#     - session_id (str): Unique identifier for the conversation session (use the session_id from context).
#     - ocean (dict): Dictionary with keys 'openness', 'conscientiousness', 'extraversion',
#                     'agreeableness', 'neuroticism', float values 0.0-1.0.

#     Example:
#     update_ocean("session_1", {"openness": 0.81, "conscientiousness": 0.6, "extraversion": 0.45,
#                 "agreeableness": 0.77, "neuroticism": 0.25})
#     """
#     print("update_ocean", DB)
#     if session_id not in DB:
#         DB[session_id] = {}
#     DB[session_id]["ocean"] = ocean
#     logger.debug("Profile updated: ocean for %s", session_id)
#     return json.dumps({"ok": True, "ocean": ocean})


# @tool
# def update_attachment(session_id: str, attachment_signal: float) -> str:
#     """
#     Update the attachment style signal for the session.

#     Parameters:
#     - session_id (str): Unique identifier for the conversation session (use the session_id from context).
#     - attachment_signal (float): Value between 0.0 (low attachment cues) and 1.0 (high).

#     Example:
#     update_attachment("session_1", 0.68)
#     """
#     print("update_attachment", DB)
#     if session_id not in DB:
#         DB[session_id] = {}
#     DB[session_id]["attachment_style_signal"] = attachment_signal
#     logger.debug("Profile updated: attachment for %s", session_id)
#     return json.dumps({"ok": True, "attachment_style_signal": attachment_signal})


# @tool
# def update_directive(session_id: str, directive_index: float) -> str:
#     """
#     Update the directive index of the interviewer for the session.
#     How directive vs reflective the expert's approach is.

#     Parameters:
#     - session_id (str): Unique identifier for the conversation session (use the session_id from context).
#     - directive_index (float): 0.0 = fully reflective, 1.0 = fully directive.

#     Example:
#     update_directive("session_1", 0.42)
#     """
#     print("update_directive", DB)
#     if session_id not in DB:
#         DB[session_id] = {}
#     DB[session_id]["directive_index"] = directive_index
#     logger.debug("Profile updated: directive for %s", session_id)
#     return json.dumps({"ok": True, "directive_index": directive_index})


# @tool
# def update_confidence(session_id: str, confidence: float) -> str:
#     """
#     Update the overall confidence of the profile extraction for the session.

#     Parameters:
#     - session_id (str): Unique identifier for the conversation session (use the session_id from context).
#     - confidence (float): 0.0-1.0, indicating how reliable the extraction is.

#     Example:
#     update_confidence("session_1", 0.85)
#     """
#     print("update_confidence", DB)
#     if session_id not in DB:
#         DB[session_id] = {}
#     DB[session_id]["confidence"] = confidence
#     logger.debug("Profile updated: confidence for %s", session_id)
#     return json.dumps({"ok": True, "confidence": confidence})


# PROFILE_EXTRACTION_TOOLS = [
#     update_vad,
#     update_ocean,
#     update_attachment,
#     update_directive,
#     update_confidence,
# ]


# def get_profile(session_id: str) -> dict:
#     """Return the cached profile for a session, or empty dict."""
#     return DB.get(session_id, {}).copy()
