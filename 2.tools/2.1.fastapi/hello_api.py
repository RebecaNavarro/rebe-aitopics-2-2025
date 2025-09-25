from fastapi import FastAPI

app = FastAPI(title="mi primera api")

estudiantes = []

@app.get("/")
def root():
    return {"message": "feliz dia del estudiante", "status": "OK"}

@app.get("/saludo")
def saludo(nombre: str, edad: int):
    return {"message": f"hola {nombre}! mucho gusto en conocerte!"}

@app.post("/estudiantes")
def crear_estudiante(nombre: str, edad: int, curso: int):
    estudiante_dic = {
        "nombre": nombre, 
        "edad": edad, 
        "curso":curso
    }
    estudiantes.append(estudiante_dic)
    return estudiante_dic

@app.get("/estudiantes")
def listar_estudiantes():
    return estudiantes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hello_api:app", reload=True, port=8008)