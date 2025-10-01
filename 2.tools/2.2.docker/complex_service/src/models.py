from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    full_name: str
    email: EmailStr
    major: str
    year: int = Field(gt=0)
    gpa: float

    class Config:
        schema_extra = {
            "example": {
                "full_name": "Juan Perez",
                "email": "juan.perez@gmail.com",
                "major": "Computer Engineering",
                "year": 2,
                "gpa": 3.5
            }
        }

