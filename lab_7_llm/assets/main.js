let inferSample = async (premise, hypothesis, result) => {
    let sample = premise.value.concat('|', hypothesis.value)
    await fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-type': 'application/json'
        },
        body: JSON.stringify({
            question: sample
        })
    }).then(res => {return res.json()})
        .then(data => {
            result.innerHTML = ''
            result.appendChild(document.createTextNode(data['infer']))
        })
}

let disableButton = (btn, premise, hypothesis) => {
    btn.disabled = !(premise.value && hypothesis.value);
}


window.onload = function() {
    const btn = document.getElementById('btn_submit');
    const premise = document.getElementById('premise');
    const hypothesis = document.getElementById('hypothesis');
    const showResults = document.getElementById('result');

    disableButton(
        btn,
        premise,
        hypothesis,
    );

    premise.addEventListener('change', () => {
        disableButton(btn, premise, hypothesis)
    });

    hypothesis.addEventListener('change', () => {
        disableButton(btn, premise, hypothesis)
    });

    btn.addEventListener('click', () => {
        inferSample(premise, hypothesis, showResults)
    })
}