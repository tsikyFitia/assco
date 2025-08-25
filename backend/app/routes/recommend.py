import torch
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.schemas.dki_schema import PastSequence
from app.dki_model.dki_model import DKTModel, recommend_next_questions, student_mastery_report, num_questions


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DKTModel(num_questions)
model.load_state_dict(torch.load("dkt_model.pt", map_location=device))
model.to(device)
model.eval()

router = APIRouter()

@router.post("/recommend")
def recommend_questions(data: PastSequence):
    seq = data.past_sequence
    top_questions = recommend_next_questions(seq, top_k=3)
    mastery = student_mastery_report(seq)

    return {
        "recommended_questions": top_questions,
        "mastery_probs": mastery["mastery_probs"].tolist(),
        "weakest_concepts": mastery["weakest_concepts"]
    }