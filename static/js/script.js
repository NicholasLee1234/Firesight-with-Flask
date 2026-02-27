const buttons = document.querySelectorAll('.nav_button');

buttons.forEach(button => {
    button.addEventListener('click', function() {
        buttons.forEach(btn => btn.classList.remove('selected'));
        this.classList.add('selected');
    });
});

function toggleElement(){
    const element = document.getElementById("control_panel");
    const toggle = document.getElementById("toggle_icon");
    if(element.style.display === "none"){
        element.style.display = "block";
        toggle.classList.add("rotate");
        toggle.classList.add("toggle_active");
    }
    else{
        element.style.display = "none";
        toggle.classList.remove("rotate");
        toggle.classList.remove("toggle_active");
    }
}