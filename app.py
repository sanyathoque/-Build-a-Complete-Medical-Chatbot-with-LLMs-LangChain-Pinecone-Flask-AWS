from flask import Flask, render_template, request

from src.rag import build_rag_chain


app = Flask(__name__)
rag_chain = build_rag_chain()


@app.get("/")
def index():
    return render_template("chat.html")


@app.post("/get")
def chat():
    question = request.form.get("msg", "").strip()

    if not question:
        return "Please ask a question."

    result = rag_chain.invoke({"input": question})
    return result["answer"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
