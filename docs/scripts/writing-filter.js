let labels = Array.from(document.querySelectorAll(".writing-filters label"));
let grid = document.getElementById("writing-grid");
let filtersContainer = document.querySelector(".writing-filters");
let allTags = labels.map((l) => l.dataset.tag);

// Show all by default
for (const tag of allTags) {
    grid.classList.add(tag);
}

labels.forEach((label) => {
    label.addEventListener("click", () => {
        let tag = label.dataset.tag;
        let wasActive = label.classList.contains("active");

        // Clear all
        labels.forEach((l) => l.classList.remove("active"));

        if (wasActive) {
            // Deselect: show all
            filtersContainer.classList.remove("has-active");
            for (const t of allTags) {
                grid.classList.add(t);
            }
        } else {
            // Select this one only
            label.classList.add("active");
            filtersContainer.classList.add("has-active");
            for (const t of allTags) {
                t === tag ? grid.classList.add(t) : grid.classList.remove(t);
            }
        }
    });
});
