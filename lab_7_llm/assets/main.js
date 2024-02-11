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
            result.appendChild(document.createTextNode(data['infer'].toUpperCase()))
        })
}

let disableButton = (btn, premise, hypothesis) => {
    btn.disabled = !(premise.value && hypothesis.value);
}


window.onload = function() {
    let btn = document.getElementById('submit');
    let premise = document.getElementById('premise');
    let hypothesis = document.getElementById('hype');
    let prediction = document.getElementById('pred');

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
        inferSample(premise, hypothesis, prediction)
    })
}
