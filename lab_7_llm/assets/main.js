let sample_inference = async (premise, hypothesis, result) => {
        let question = premise.value.concat("|", hypothesis.value)
        await fetch("/infer", {
            method: "POST",
            headers: {
                "Content-type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        }).then(response => {
            return response.json()
        }).then(response => {
            result.innerHTML = ""
            result.appendChild(document.createTextNode(response["infer"].toUpperCase()))
        })
    };

window.onload = function(){
    const premise = document.getElementById("premise")
    const hypothesis = document.getElementById("hypothesis")
    const button = document.getElementById("button")
    const result = document.getElementById("result")
    button.addEventListener("click", ()=>{sample_inference(premise, hypothesis, result)})
};
