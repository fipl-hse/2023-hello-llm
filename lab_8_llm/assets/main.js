let sample_inference = async (question, result) => {
        const response = await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        })
        let results_infer = await response.json();
        console.log(results_infer['infer'])
        result.innerHTML = results_infer['infer'];
        // result.appendChild(document.createTextNode(results_infer['infer'.toUpperCase()]))
}

window.onload = function(){
    const question = document.getElementById("question")
    const button = document.getElementById("button")
    const result = document.getElementById("result")
    button.addEventListener("click", ()=>{sample_inference(question, result)})
};