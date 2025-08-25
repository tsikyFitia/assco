from pydantic import BaseModel
from typing import List


class PastSequence(BaseModel):
    past_sequence: List[List[int]]  # [[question_id, correct_flag], ...]