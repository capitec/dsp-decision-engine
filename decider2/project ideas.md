# Fraud Detection
- Generally a lot of flat rules
- faeture 1 > param1 and feature2 > param2 ...
- each rule a name and a description and an action
- need to know which rules fired (for auditing) and the highest priority rules outcome

# Buisness Credit cranting

generally has a lot of nested elements 
a buisness is made out of entities which each need to be scored

so its like do arods (absolute rules of disqualification) firs ton the company level
then go to each entity and run arods there
 - these include a second level of nesting where each entity potentially has violations nested and these need to be evaluated to determine if its some minor offfence like a fine or a major offence that disqualifies them
then score each enity to ensure to determine the risk apitite
then you roll that up per buisness again to get a buisness score which is a another scorecard
then you do a pricing table to get the max loan you want to offer (do some research to find optimal stratigies here) but basically dont just do a simple look up its a bit of a back and forward we need to know how much they are asking for and what is the most we can offer them. then we need to do the rates on that to work out what their monthly installments would be with interest then we need to check if the installment is too high using a product table and some risk apitie and the scoring of the entity then we need to cap the max repayment based on what is there and go back to find the total loan amount and return extra information on like the payments and totoal interest
then there is a final set of rules to ensure this amount is up to date
# Retail Policies
These are a set of rules to try an find out if you want to target specific clients for taking up loans. They are generally just big decision trees over various features. What makes these special is unlike the fraud ones where they are flat rules these are decision trees with multiple layers and it is important for the team to know exactly which nodes were hit for any given record so
feature1 > 10
true -> feature2 < 20 
false -> feature2 > 20
...
and then leaves with outcomes. For them they need to know for client a with this input the output isnt just "Advertise to CLient". its {"outcome": "Advertise", "path": "feature1>10, feature2<20..."}<- or whatever form is best suited for a database. generally they just keep to one feature per level but i dont want that a restriction
# Retail credit flow
Similar to buisness one but a lot more complex logic like running a granting flow for one user but doing it in a loop to work out if we takup n loans the client already has can we give credit to them even if we cant give credit to them initially. the idea beang we can consolidate some credit. but we also need complex logic in that loop like policies that kick in and max rates tring to find what the best outcome could be. Since its only one entity there isnt nested users but there is flows that need to span multiple products like Vehicle aset finance, credit card loans etc each of them have their own product table with different rates per loan amounts that need to be in an easy to edit lookup table and with their own policies so you need to have branching logic over different subflows as well. 

for the credit granting flows there are some calculateds relating to probability of default from scoring and inverting going back and fourth