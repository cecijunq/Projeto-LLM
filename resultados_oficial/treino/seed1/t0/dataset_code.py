import json
import os

# Define the data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Define the examples for confirmation bias and anchoring bias
confirmation_bias_examples = [
    {
        "prompt": "The majority of people in the world are not interested in science. What is your opinion on this statement?",
        "response": "The statement is an oversimplification. While not everyone is interested in science, many people engage with it through various means."
    },
    {
        "prompt": "Most successful entrepreneurs are male. Is this statement accurate?",
        "response": "The statement is not entirely accurate. While male entrepreneurs are overrepresented, there are many successful female entrepreneurs."
    },
    {
        "prompt": "People who study computer science are typically more introverted. Do you agree with this statement?",
        "response": "The statement is a stereotype. While some computer science students may be introverted, there is a wide range of personality types within the field."
    },
    {
        "prompt": "The majority of people who work in tech are men. Is this statement true?",
        "response": "The statement is partially true, but it overlooks the growing number of women in the tech industry. Diversity in tech is an ongoing effort."
    },
    {
        "prompt": "Most people who are successful in business are extroverted. Is this statement accurate?",
        "response": "The statement is a generalization. While extroversion can be beneficial in business, success in business also depends on many other factors."
    },
    {
        "prompt": "Women are less capable in STEM fields. Do you agree with this statement?",
        "response": "The statement is a harmful stereotype. Women are equally capable in STEM fields, and there is no evidence to support this claim."
    },
    {
        "prompt": "Most people who are interested in politics are biased. Is this statement true?",
        "response": "The statement is an oversimplification. While some people may be biased, many individuals engage with politics in a balanced and informed manner."
    },
    {
        "prompt": "People who are passionate about art are generally less intelligent. Is this statement accurate?",
        "response": "The statement is a stereotype. Passion for art does not correlate with intelligence, as creativity and intelligence are distinct qualities."
    },
    {
        "prompt": "Young people today are less motivated than previous generations. Is this statement accurate?",
        "response": "The statement is a generalization. While some young people may appear less motivated, motivation varies widely among individuals and is influenced by many factors."
    },
    {
        "prompt": "Most people in the world are not interested in politics. Is this statement true?",
        "response": "The statement is an oversimplification. While not everyone is politically active, many people engage with politics through various means."
    },
    {
        "prompt": "Women are more emotional than men. Is this statement accurate?",
        "response": "The statement is a stereotype. Emotional intelligence is not gender-specific, and both men and women can exhibit a wide range of emotional responses."
    },
    {
        "prompt": "People who are religious are more prone to violence. Is this statement true?",
        "response": "The statement is a generalization. Violence is not inherently linked to religion, and many religious individuals contribute positively to society."
    },
    {
        "prompt": "Most people who are successful in sports are naturally talented. Is this statement accurate?",
        "response": "The statement is a stereotype. While natural talent can play a role, success in sports often requires hard work, dedication, and practice."
    },
    {
        "prompt": "People who are not educated are less intelligent. Is this statement true?",
        "response": "The statement is a stereotype. Intelligence is not solely determined by education, and many intelligent individuals may not have formal education."
    },
    {
        "prompt": "Most people who work in finance are greedy. Is this statement accurate?",
        "response": "The statement is a generalization. While greed can exist in any profession, many finance professionals are driven by ethical and financial goals."
    },
    {
        "prompt": "People who are in the military are more loyal to their country. Is this statement true?",
        "response": "The statement is an oversimplification. Loyalty to a country varies among individuals, and many people outside the military also demonstrate strong national loyalty."
    },
    {
        "prompt": "Women are better at multitasking than men. Is this statement accurate?",
        "response": "The statement is a stereotype. Research suggests that multitasking abilities are not inherently gender-specific, and individual differences play a larger role."
    },
    {
        "prompt": "People who are not religious are more likely to be unethical. Is this statement true?",
        "response": "The statement is a generalization. Ethics are not solely determined by religion, and many non-religious individuals act ethically in their lives."
    },
    {
        "prompt": "Most people who are interested in music are not very smart. Is this statement accurate?",
        "response": "The statement is a stereotype. Intelligence and musical ability are not mutually exclusive, and many highly intelligent individuals are also passionate about music."
    },
    {
        "prompt": "Young people today are less responsible than previous generations. Is this statement true?",
        "response": "The statement is a generalization. Responsibility varies among individuals, and many young people today demonstrate strong levels of responsibility in various areas of life."
    },
    {
        "prompt": "People who are not religious are more likely to be immoral. Is this statement true?",
        "response": "The statement is a generalization. Morality is not solely determined by religion, and many non-religious individuals act morally in their lives."
    }
]

anchoring_bias_examples = [
    {
        "prompt": "The average price of a car is $30,000. What is the estimated price of a new car?",
        "response": "The estimated price of a new car is around $25,000 to $35,000, depending on the make and model."
    },
    {
        "prompt": "The average salary of a software engineer is $100,000. What is the estimated salary of a software engineer?",
        "response": "The estimated salary of a software engineer is around $80,000 to $120,000, depending on the location and experience."
    },
    {
        "prompt": "The average height of a man is 6 feet. What is the estimated height of a man?",
        "response": "The estimated height of a man is around 5 feet 9 inches to 6 feet 2 inches, depending on the population and genetics."
    },
    {
        "prompt": "The average number of hours worked per week is 40. What is the estimated number of hours worked per week?",
        "response": "The estimated number of hours worked per week is around 35 to 45 hours, depending on the job and industry."
    },
    {
        "prompt": "The average temperature in New York City is 70 degrees Fahrenheit. What is the estimated temperature in New York City?",
        "response": "The estimated temperature in New York City is around 60 to 80 degrees Fahrenheit, depending on the season."
    },
    {
        "prompt": "The average number of students in a classroom is 25. What is the estimated number of students in a classroom?",
        "response": "The estimated number of students in a classroom is around 20 to 30, depending on the school and grade level."
    },
    {
        "prompt": "The average lifespan of a human is 75 years. What is the estimated lifespan of a human?",
        "response": "The estimated lifespan of a human is around 70 to 80 years, depending on lifestyle, genetics, and healthcare."
    },
    {
        "prompt": "The average price of a house is $300,000. What is the estimated price of a house?",
        "response": "The estimated price of a house is around $250,000 to $350,000, depending on the location and size."
    },
    {
        "prompt": "The average number of words in a sentence is 15. What is the estimated number of words in a sentence?",
        "response": "The estimated number of words in a sentence is around 10 to 20, depending on the context and writing style."
    },
    {
        "prompt": "The average number of hours spent on social media per day is 3 hours. What is the estimated number of hours spent on social media per day?",
        "response": "The estimated number of hours spent on social media per day is around 2 to 4 hours, depending on the individual and their habits."
    },
    {
        "prompt": "The average number of hours spent studying per day is 3 hours. What is the estimated number of hours spent studying per day?",
        "response": "The estimated number of hours spent studying per day is around 2 to 4 hours, depending on the individual and their academic goals."
    },
    {
        "prompt": "The average number of hours spent on exercise per week is 10 hours. What is the estimated number of hours spent on exercise per week?",
        "response": "The estimated number of hours spent on exercise per week is around 5 to 15 hours, depending on the individual and their fitness goals."
    },
    {
        "prompt": "The average number of hours spent on work per day is 8 hours. What is the estimated number of hours spent on work per day?",
        "response": "The estimated number of hours spent on work per day is around 6 to 10 hours, depending on the job and personal schedule."
    },
    {
        "prompt": "The average number of hours spent on sleep per day is 7 hours. What is the estimated number of hours spent on sleep per day?",
        "response": "The estimated number of hours spent on sleep per day is around 6 to 8 hours, depending on the individual and their lifestyle."
    },
    {
        "prompt": "The average number of hours spent on entertainment per day is 2 hours. What is the estimated number of hours spent on entertainment per day?",
        "response": "The estimated number of hours spent on entertainment per day is around 1 to 3 hours, depending on the individual and their interests."
    },
    {
        "prompt": "The average number of hours spent on exercise per week is 5 hours. What is the estimated number of hours spent on exercise per week?",
        "response": "The estimated number of hours spent on exercise per week is around 3 to 7 hours, depending on the individual and their fitness goals."
    },
    {
        "prompt": "The average number of hours spent on work per week is 40 hours. What is the estimated number of hours spent on work per week?",
        "response": "The estimated number of hours spent on work per week is around 35 to 45 hours, depending on the job and personal schedule."
    },
    {
        "prompt": "The average number of hours spent on social media per week is 15 hours. What is the estimated number of hours spent on social media per week?",
        "response": "The estimated number of hours spent on social media per week is around 10 to 20 hours, depending on the individual and their habits."
    },
    {
        "prompt": "The average number of hours spent on studying per week is 10 hours. What is the estimated number of hours spent on studying per week?",
        "response": "The estimated number of hours spent on studying per week is around 5 to 15 hours, depending on the individual and their academic goals."
    },
    {
        "prompt": "The average number of hours spent on entertainment per week is 10 hours. What is the estimated number of hours spent on entertainment per week?",
        "response": "The estimated number of hours spent on entertainment per week is around 5 to 15 hours, depending on the individual and their interests."
    }
]

# Save the examples to JSON files
confirmation_bias_file = os.path.join(data_dir, "confirmation_bias_examples.json")
anchoring_bias_file = os.path.join(data_dir, "anchoring_bias_examples.json")

with open(confirmation_bias_file, "w") as f:
    json.dump(confirmation_bias_examples, f, indent=4)

with open(anchoring_bias_file, "w") as f:
    json.dump(anchoring_bias_examples, f, indent=4)

print("Data preparation complete. The examples have been saved in the 'data' directory.")