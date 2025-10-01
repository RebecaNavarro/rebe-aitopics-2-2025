from fastapi import APIRouter, Body, HTTPException
from fastapi.encoders import jsonable_encoder

from database import (
    add_student,
    retrieve_student,
    retrieve_students,
)
from models import Student

router = APIRouter()

@router.post("/")
async def create_student(student: Student):
    student = jsonable_encoder(student)
    new_student = await add_student(student)
    return new_student


@router.get("/", response_description="Students retrieved")
async def get_students():
    students = await retrieve_students()
    if students:
        return students
    return []


@router.get("/{id}", response_description="Student data retrieved")
async def get_student_data(id):
    student = await retrieve_student(id)
    if student:
        return student
    raise HTTPException(404, detail="student not found")