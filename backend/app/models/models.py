from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import date, datetime


class Institution(BaseModel):
    name: str
    address: str
    email: EmailStr
    phone: str
    created_at: Optional[datetime]

    


class User(BaseModel):
    last_name: str
    first_name: str
    email: EmailStr
    password_hash: str
    birth_date: Optional[date]
    created_at: Optional[datetime]
    role: str
    institution_id: Optional[str]

    


class Guardian(BaseModel):
    user_id: str

    


class Level(BaseModel):
    name: str
    cycle: str
    average_age: int
    description: Optional[str]
    institution_id: str

    


class Student(BaseModel):
    user_id: str
    level_id: str
    guardian_id: str

    


class Teacher(BaseModel):
    user_id: str

    


class Subject(BaseModel):
    name: str
    code: str

    


class Program(BaseModel):
    level_id: str
    subject_id: str

    


class Teaching(BaseModel):
    teacher_id: str
    subject_id: str
    level_id: str

    


class Content(BaseModel):
    title: str
    type: str
    format: str
    file_id: str
    uploaded_at: Optional[datetime]
    validated_by_admin: bool
    subject_id: str
    level_id: str
    teacher_id: str
    institution_id: str

    


class Exercise(BaseModel):
    title: str
    evaluation_type: str
    author: str
    questions: List[str]
    created_at: Optional[datetime]
    subject_id: str
    level_id: str
    creator_id: str
    institution_id: str

    


class Submission(BaseModel):
    evaluation_type: str
    grade: Optional[float]
    submitted_at: Optional[datetime]
    validated_by_teacher: bool
    auto_grade: Optional[float]
    auto_feedback: Optional[str]
    manual_grade: Optional[float]
    manual_teacher_id: Optional[str]
    manual_comment: Optional[str]
    student_id: str
    exercise_id: str

    


class Progress(BaseModel):
    date: Optional[date]
    avg_exercise_grade: Optional[float]
    avg_quiz_grade: Optional[float]
    avg_exam_grade: Optional[float]
    exercise_count: int
    quiz_count: int
    exam_count: int
    weak_topics: List[str]
    strong_topics: List[str]
    overall_average: Optional[float]
    exercise_average: Optional[float]
    quiz_average: Optional[float]
    exam_average: Optional[float]
    student_id: str
    subject_id: str
    level_id: str

    


class AIRecommendation(BaseModel):
    type: str
    reason: Optional[str]
    status: str
    recommended_at: Optional[datetime]
    student_id: str
    content_id: str

    
