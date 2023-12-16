import re
from nltk.tokenize import blankline_tokenize
text = "Project section:\n\nThis section is about a project that I worked on to develop a new mobile app for our company. The app was designed to make it easier for our customers to manage their accounts and to make purchases. The app was launched on time and within budget, and it has been well-received by our customers.\n\nThis section is about a team project that I worked on to design and build a new website for our company. The website was designed to be user-friendly and informative, and to showcase our products and services. The website was launched on time and within budget, and it has resulted in an increase in traffic and leads.\n\nNon-project section:\n\nThis section is about my company's mission and values. Our mission is to provide our customers with the best possible products and services. Our values are integrity, customer service, and innovation.\n\nThis section is about my professional goals and aspirations. My goal is to become a leading expert in my field and to make a significant contribution to my profession."

# Define the list of keywords
keywords = ["Project", "Client", "Experience", "Scope", "Achievement", "Result", "Summary", "Client", "Customer", "Task", "Assignment", "Responsibilities", "Accomplishments", "Deliverables", "Tasks", "Challenges", "Goals", "Objectives", "Initiatives", "Outcomes", "Successes", "Learnings", "Approach", "Methodology", "Solution", "Challenges", "Milestones", "Projects", "Clients", "Experiences", "Scopes", "Achievements", "Results", "Summaries", "Clients", "Customers", "Tasks", "Assignments"]

Project = []

def Count_Paragraph():
    processed_paragraphs = []

    #   checks for no. of paragraph                        
    remove_bullet = text.split('•')
    data_without_bullet = ''.join(remove_bullet)
    remove_hyphen_bullet = data_without_bullet.split('-')
    data_without_bullet_hyphen = ''.join(remove_hyphen_bullet)
    pattern = r' {2,}|\n'
    isolated_paragraphs = re.split(pattern, data_without_bullet_hyphen)

    for paragraph in isolated_paragraphs:
        words = paragraph.split()
        current_paragraph = ""
        next_paragraph = ""
        
        for word in words:
            current_paragraph += next_paragraph
            next_paragraph = word

            if len(word) <= 5:
                next_paragraph = current_paragraph
                current_paragraph = ""
            else:
                processed_paragraphs.append(current_paragraph)
                current_paragraph = ""

        # Check if any word in the paragraph is in the list of keywords
        if any(word in keywords for word in words):
            Project.append(processed_paragraphs)

        processed_paragraphs.append(next_paragraph)
    
        
project= ""
Count_Paragraph()
for idx, project_paragraph in enumerate(Project):
        project=' '.join(project_paragraph)
        


print(project)