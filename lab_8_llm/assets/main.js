
const submitButton = document.getElementById("submit");
const questionInput = document.getElementById("question");
const responseDiv = document.getElementById("response");

submitButton.addEventListener("click", async function() {
    responseDiv.textContent = "Awaiting a response..."
    const question = questionInput.value;
    const response = await fetch("/infer", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
    });

    const data = await response.json();
    responseDiv.textContent = data.infer;
});
