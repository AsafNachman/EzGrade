# Project Proposal: EzGrade – AI-Powered Syllabus-to-Exam Generator

**Team Members:** Asaf, Roei, Ilai, Eliran, Yuval, Salah

## 1. The User and the Problem
**Users:** University and college students.
**The Problem:** Students often struggle to find relevant practice exams for their specific courses, especially when a syllabus changes or past exams are unavailable. Passive reading is ineffective, and students lack a tool to actively test their knowledge on specific syllabus topics to identify their weaknesses before the real exam.

## 2. Today's Workaround
Currently, students rely on searching for old exams in student drives (which may be outdated), manually creating flashcards on platforms like Quizlet (which is time-consuming), or asking ChatGPT for general questions (which often results in broad, unstructured, or out-of-scope questions).

## 3. Competitors
- **General LLMs (ChatGPT/Claude):** Require complex prompt engineering and don't provide a structured, interactive exam interface or progress tracking.
- **Quizlet / Anki:** Excellent for memorization but require manual input and don't automatically generate full exams directly from a syllabus document.

## 4. Why Switch?
EzGrade automates the entire process. Instead of writing prompts or manually creating cards, a student simply uploads their course syllabus, selects the specific topics they want to study, and instantly gets a structured, interactive Multiple-Choice Question (MCQ) exam tailored to their requested difficulty level.

## 5. MVP (Minimum Viable Product)
The MVP will focus on a clear scope of 3 user stories:
1. **Syllabus Parsing:** As a student, I want to upload a syllabus (PDF/Text) so the system can extract a structured list of topics for me to choose from.
2. **Central User Story - Exam Generation:** As a student, I want to select specific topics and a difficulty level (Easy/Medium/Hard) to generate a customized 100% MCQ practice exam.
3. **Grading & Feedback:** As a student, I want to submit my exam to receive an immediate grade, detailed explanations for my mistakes, and a breakdown of my weak topics.

*What is left out:* User authentication, long-term saving of score history across multiple sessions, and non-MCQ open-ended questions.

## 6. Success Metric
1. The system successfully extracts at least 80% of the main topics from 5 different standard introductory course syllabi.
2. Generating a valid, interactive 10-question MCQ exam takes under 30 seconds.

## 7. Risks
**AI Hallucinations and Inaccuracies:** The AI might generate questions that are outside the scope of the selected topic or provide multiple correct answers for a single question. 
*Mitigation:* We will use strict prompt engineering, JSON schema validation (e.g., Pydantic), and a validation step to ensure each question has exactly one correct answer and matches the requested topic.

## 8. Role of AI Agents
Yes, AI plays a core, required role in this system: It is used to parse the unstructured syllabus text into topics, generate the syllabus-specific questions and distractors, and provide pedagogical explanations for incorrect answers.
