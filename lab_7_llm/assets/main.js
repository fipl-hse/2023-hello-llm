document.addEventListener("DOMContentLoaded", function() {
    const sourceText = document.getElementById("source");
    const predictionText = document.getElementById("prediction");
    const sendButton = document.getElementById("send");

    sendButton.addEventListener("click", async function() {
        const question = sourceText.value;
        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        predictionText.textContent = data.infer;
    });
});