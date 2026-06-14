export interface Book {
	title: string
	author: string
	cover?: string // filename in /public/images/books/, omit for a text-only card
}

export interface BookGroup {
	label: string
	books: Book[]
}

export const bookGroups: BookGroup[] = [
	{
		label: '2025',
		books: [
			{ title: 'The Idea Factory', author: 'Pepper White', cover: 'idea-factory.jpg' },
			{ title: 'Stories of Your Life and Others', author: 'Ted Chiang', cover: 'stories-of-your-life.jpg' },
			{ title: 'Deep Work', author: 'Cal Newport', cover: 'deep-work.jpg' },
			{ title: 'A Theory of Fun for Game Design', author: 'Raph Koster', cover: 'theory-of-fun.jpg' },
			{ title: 'What I Talk About When I Talk About Running', author: 'Haruki Murakami', cover: 'talk-about-running.jpg' }
		]
	},
	{
		label: '2024',
		books: [
			{ title: 'Situational Awareness', author: 'Leopold Aschenbrenner' },
			{ title: 'The Little Book of Deep Learning', author: 'François Fleuret' },
			{ title: 'Meta Learning', author: 'Radek Osmulski' }
		]
	},
	{
		label: '2023',
		books: [
			{ title: 'The Metamorphosis of Prime Intellect', author: 'Roger Williams', cover: 'prime-intellect.jpg' },
			{ title: 'The Usefulness of Useless Knowledge', author: 'Abraham Flexner', cover: 'useless-knowledge.jpg' },
			{ title: 'Memories of a Theoretical Physicist', author: 'Joseph Polchinski' }
		]
	},
	{
		label: '2022',
		books: [
			{ title: 'Lost in Math', author: 'Sabine Hossenfelder', cover: 'lost-in-math.jpg' },
			{ title: 'Not Even Wrong', author: 'Peter Woit', cover: 'not-even-wrong.jpg' },
			{ title: 'The Trouble with Physics', author: 'Lee Smolin', cover: 'trouble-with-physics.jpg' },
			{ title: 'The Foundation', author: 'Isaac Asimov', cover: 'foundation.jpg' },
			{ title: 'Artemis', author: 'Andy Weir', cover: 'artemis.jpg' },
			{ title: 'Project Hail Mary', author: 'Andy Weir', cover: 'project-hail-mary.jpg' }
		]
	},
	{
		label: 'Started but not finished',
		books: [
			{ title: 'Brainstorms', author: 'Daniel C. Dennett', cover: 'brainstorms.jpg' },
			{ title: 'Dune', author: 'Frank Herbert', cover: 'dune.jpg' },
			{ title: 'Tiempos Recios', author: 'Mario Vargas Llosa', cover: 'tiempos-recios.jpg' },
			{ title: 'Maker of Patterns', author: 'Freeman Dyson' },
			{ title: '1984', author: 'George Orwell', cover: '1984.jpg' },
			{ title: "Hitler's First Hundred Days", author: 'Peter Fritzsche', cover: 'hitlers-first-hundred-days.jpg' },
			{ title: 'Androids', author: 'Chet Haase', cover: 'androids.jpg' },
			{ title: 'Countdown to Zero Day', author: 'Kim Zetter', cover: 'countdown-zero-day.jpg' },
			{ title: 'Just for Fun', author: 'Linus Torvalds', cover: 'just-for-fun.jpg' },
			{ title: 'Alien Oceans', author: 'Kevin Hand', cover: 'alien-oceans.jpg' },
			{ title: 'A Guide to Competitive Programming', author: 'Antti Laaksonen', cover: 'guide-cp.jpg' },
			{ title: 'The Science of Interstellar', author: 'Kip Thorne', cover: 'science-interstellar.jpg' },
			{ title: 'Feynman Lectures', author: 'Richard Feynman', cover: 'feynman-lectures.jpg' },
			{ title: "Surely You're Joking, Mr. Feynman", author: 'Richard Feynman', cover: 'surely-joking-feynman.jpg' },
			{ title: 'QED', author: 'Richard Feynman', cover: 'qed.jpg' },
			{ title: 'Atomic Habits', author: 'James Clear', cover: 'atomic-habits.jpg' },
			{ title: 'Solving Mathematical Problems', author: 'Terence Tao', cover: 'solving-math-problems.jpg' },
			{ title: 'How to Solve It', author: 'George Pólya', cover: 'how-to-solve-it.jpg' },
			{ title: 'The Birth of a Theorem', author: 'Cédric Villani', cover: 'birth-of-theorem.jpg' },
			{ title: 'Quantum Computing Since Democritus', author: 'Scott Aaronson', cover: 'quantum-computing-democritus.jpg' },
			{ title: 'J. A. Balseiro: Crónica de una Ilusión', author: 'Arturo López Dávalos', cover: 'balseiro.jpg' }
		]
	}
]
