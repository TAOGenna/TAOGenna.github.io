let checkboxes = Array.from(document.querySelectorAll(".writing-filters input[type='checkbox']"));
let grid = document.getElementById("writing-grid");

for (const cb of checkboxes) {
    grid.classList.add(cb.id);
}

checkboxes.forEach((cb) => {
    cb.checked = false;
    cb.addEventListener("change", filterPosts);
});

function filterPosts() {
    let checked = Array.from(document.querySelectorAll(".writing-filters input[type='checkbox']:checked"));
    if (checked.length === 0) {
        for (const cb of checkboxes) {
            grid.classList.add(cb.id);
        }
    } else {
        for (const cb of checkboxes) {
            cb.checked ? grid.classList.add(cb.id) : grid.classList.remove(cb.id);
        }
    }
}
