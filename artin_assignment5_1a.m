close all
clear
clc

load data_SFcrime.mat

%% Part 1
n_data = size(Category,1);

hours = hour(Dates); %extracting hours
hours_s = sparse(1:n_data, hours + 1, ones(n_data,1)); %1-hot vector

keySet =   {'Sunday', 'Monday', 'Tuesday', 'Wednesday',...
    'Thursday', 'Friday', 'Saturday'};
valueSet = [1,2,3,4,5,6,7];
mapObj = containers.Map(keySet,valueSet);
days = zeros(n_data,1);
for i=1:n_data
    days(i)=mapObj(char(DayOfWeek(i))); %extracting the days of the week
end
days_s = sparse(1:n_data, days, ones(n_data,1)); %1-hot vector

[pd_unique,~,pd_id] = unique(PdDistrict); %Giving an ID to each PD-District
pd_s = sparse(1:n_data, pd_id, ones(n_data,1)); %1-hot vector

data_s = [hours_s, days_s, pd_s]; %Preprocessed Data

%% Part 2
figure
histogram(hours,24);
title('Histogram of Hour')
xlabel('Hour of Day')
ylabel('Occurance')
savefig('5_1a_Hour.fig');
saveas(gcf,'5_1a_Hour.jpg');

figure
histogram(days);
xticks(1:7)
xticklabels({'Sun','Mon','Tue','Wed','Thu','Fri','Sat'})
title('Histogram of Day of Week')
xlabel('Day of Week (Sun to Sat)')
ylabel('Occurance')
savefig('5_1a_Days.fig');
saveas(gcf,'5_1a_Days.jpg');

figure
histogram(categorical(PdDistrict));
title('Histogram of Police Department District')
xlabel('Police Department District')
ylabel('Occurance')
savefig('5_1a_PD.fig');
saveas(gcf,'5_1a_PD.jpg');

figure
histogram(categorical(Category));
title('Histogram of Category')
xlabel('Category')
ylabel('Occurance')
savefig('5_1a_Category.fig');
saveas(gcf,'5_1a_Category.jpg');

%% Part 3
[cat_unique,~,cat_id] = unique(Category);
n_cat = size(cat_unique,1); %number of categories
max_hour_crime = zeros(n_cat,1); %most likely hour of crime
for i = 1:n_cat
    max_hour_crime(i) = mode(hours(cat_id==i));
end
table(cat_unique, max_hour_crime,'VariableNames',{'Crimes','Hour'})

%% Part 4
n_pd = size(pd_unique,1); %number of PD Districts
max_crime_pd = zeros(n_pd,1); %most likely crime of PD
for i = 1:n_pd
    max_crime_pd(i) = mode(cat_id(pd_id==i));
end
table(pd_unique, cat_unique(max_crime_pd), 'VariableNames',...
    {'PD_District', 'Crime'})

%% Part 5
figure %Drug/Narcotic
scatter(X(cat_id==8),Y(cat_id==8),4,'r','filled','MarkerFaceAlpha',.2)
title('Locations of Drug/Narcotic crime')
xlabel('Longitude (x-coordinate)')
ylabel('Latitude (y-coordinate)')
plot_google_map
savefig('5_1a_map_Drug.fig');
saveas(gcf,'5_1a_map_Drug.jpg');

figure %Larceny/Theft
scatter(X(cat_id==17 & Y<=50),Y(cat_id==17 & Y<=50),4,'b','filled','MarkerFaceAlpha',.2)
title('Locations of Larceny/Theft crime')
xlabel('Longitude (x-coordinate)')
ylabel('Latitude (y-coordinate)')
plot_google_map
savefig('5_1a_map_Larceny.fig');
saveas(gcf,'5_1a_map_Larceny.jpg');
 
figure %Robbery
scatter(X(cat_id==26 & Y<=50),Y(cat_id==26 & Y<=50),4,'black','filled','MarkerFaceAlpha',.2)
title('Locations of Robbery crime')
xlabel('Longitude (x-coordinate)')
ylabel('Latitude (y-coordinate)')
plot_google_map
savefig('5_1a_map_Robbery.fig');
saveas(gcf,'5_1a_map_Robbery.jpg');

save('5_1a_results');
save('data_sparse','data_s','n_data','cat_unique','cat_id','Category');