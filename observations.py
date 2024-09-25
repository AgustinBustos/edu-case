obs_dict1={
    'default':'',
    'Questions 1 and 2':"Question 1: A priori, it is very dificult to hipothesize about the main drivers that students take into acount when deciding a career, also, we have to make sure that the client can influence those variables, for example, if we determine that the students go on a specific career path due to their parents paths, that information wouldn't be usefull at all since the client cannot do anything about it. So taking that out of the picture, lets start with the 'in the box' thinking, price, marketing, trade and distribution are the usual old boring variables that permeate businesses, so the classic CPG study on those variables is fundamental. For example, maybe just decreasing the price, or adjusting the price student is more usefull that one might think at first glance. There are infinite ways of changing prices, for example, a scholarship system is just a consumer discrimination price strategy. Regarding marketing, focusing on the university rankings is crucial for the consumer choice. \n Now come the 'out of the box' hipothesis, brainstorming requires more people, but maybe an idea would be to open new careers. \n\n  Now that we recognize all the possible levers, the information must be scraped in order to correctly actionize on them; following the previous examples we would collect time series of our and our competitors investment in marketing and rankings, geographical distribution, prices, percent of scholarships types of careers available will be important, obviously, there will be an increasing quantity of ideas from what to collect, but this should be a good starting point, understand the market.",

   
}
obs_dict2={
    'default':'',
    'Questions 1 and 2':"Question 2: The main available columns to take into account will be 'Institucion', 'Año', 'Curso', 'Probabilidad UIN', 'Carrera 1', 'Carrera 2', 'Carrera 3', 'Universidad 1', 'Universidad 2', 'Universidad 3'. The columns that we want to influence are 'Universidad 1', 'Universidad 2', 'Universidad 3', 'Probabilidad UIN', so the KPIs will be constructed from these columns, for simplification, i just used 1 KPI that being a dummy variable that is 1 when the 'Universidad 1' column is UNAM or UIN (i just asumed that the client is owner of these 2 universities) and 0 if it is other value. Now we want to understand how the other columns influence our KPI, first i preprocessed a little bit the other columns, using the openai api, i detected the probable gender of the students from their name, and clustered the 'Carrera 1' column so that it is more understandable, i once again dropped 'Carrera 2' and 'Carrera 3' due to time considerations for simplification. Intersecting vizualizations and predictive models is the best way to deeply understand how the underliying data affects the KPIs.",

   
}