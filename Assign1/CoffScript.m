% Script to calculate the Pearson Correlation Coffiecient and store it in
% 15X15 matrix

pearsonCorelationCoeff = zeros(15,15);
for i = 1:15
    for j = 1:15
        c = cov(SkillCraft1Dataset1(:,i),SkillCraft1Dataset1(:,j));
        std1 = std(SkillCraft1Dataset1(:,i));
        std2 = std(SkillCraft1Dataset1(:,j));
        tempPCC = c(2) / (std1*std2);
        pearsonCorelationCoeff(i,j) = tempPCC;
    end
end

