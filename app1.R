
library(plyr)
library(shiny)
library(caret)


ui <- fluidPage(
    titlePanel("Airbnb Room Classification"),
    sidebarLayout(
        sidebarPanel(
            radioButtons("neighbourhood_group", "Neighbourhood Group",
                         c("Bronx", "Brooklyn", "Manhattan","Queens","Staten Island"),
                         conditionalPanel(condition = "input.neigbourhood_group.includes('Bronx')"
                         )),
            
            conditionalPanel(condition = 'input.neighbourhood_group=="Bronx"',selectInput("neighbourhood","Neighbourhood",choices = c("Mott Haven","Williamsbridge","Longwood","Van Nest","Pelham Gardens","Allerton"))),
            
            conditionalPanel(condition = 'input.neighbourhood_group=="Brooklyn"',selectInput("neighbourhood","Neighbourhood",choices = c("Williamsburg","Greenpoint","Prospect Heights","Flatbush","Clinton Hill","Fort Greene"))),
            
            conditionalPanel(condition = 'input.neighbourhood_group=="Manhattan"',selectInput("neighbourhood","Neighbourhood",choices = c("Financial District","Chelsea","Tribeca","Greenwich Village","Hell's Kitchen","Midtown"))),
            
            conditionalPanel(condition = 'input.neighbourhood_group=="Queens"',selectInput("neighbourhood","Neighbourhood",choices = c("Astoria","Jamaica","Long Island City","Flushing","Ridgewood","Far Rock Away"))),
            
            conditionalPanel(condition = 'input.neighbourhood_group=="Staten Island"',selectInput("neighbourhood","Neighbourhood",choices = c("Todt Hill","Stapleton","Port Richmond","Shore Acres","South Beach","Willowbrook"))),
            
            textInput("lat","Latitude",placeholder = "Between 40.50 and 40.90"),
            
            textInput("long","Longitude",placeholder = "Between -73.70 and -74.20"),
            
            selectInput("room_type","Room Type",selected = NULL,choices = c("Entire home/apt","Shared room","Private room")),
            
            textInput("nights","Number of nights",placeholder = "Between 1 and 1000"),
            
            textInput("reviews","Number of reviews",placeholder = "Between 0 and 600"),
            
            textInput("reviews_pm","Reviews per month",placeholder = "Between 0 and 58"),
            
            textInput("host","Host Listings Count",placeholder = "Between 1 and 300"),
            
            textInput("available","Availability",placeholder = "Out of 365"),
            
            radioButtons("model_selector", "Model",
                         choices = c("Support Vector Machine", "Random Forest Classifier", "Decision Tree Classifier"),
                         selected = "")
        ),
        mainPanel(
            textOutput(outputId = 'model'),verbatimTextOutput(outputId = 'pred')
        )
    )
)

svm_prediction<- function(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10)
{
    df<- data.frame(neighbourhood_group=as.factor(c1),neighbourhood=as.factor(c2),latitude=as.numeric(c3),longitude=as.numeric(c4),room_type=as.factor(c5),minimum_nights=as.numeric(c6),number_of_reviews=as.numeric(c7),reviews_per_month=as.numeric(c8),calculated_host_listings_count=as.numeric(c9),availability_365=as.numeric(c10))  
    svm_model<-readRDS("svm_model_tuned.rds")
    pred<-predict(svm_model,df)
    return(pred)
}

tree_prediction<- function(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10)
{
    df<- data.frame(neighbourhood_group=as.factor(c1),neighbourhood=as.factor(c2),latitude=as.numeric(c3),longitude=as.numeric(c4),room_type=as.factor(c5),minimum_nights=as.numeric(c6),number_of_reviews=as.numeric(c7),reviews_per_month=as.numeric(c8),calculated_host_listings_count=as.numeric(c9),availability_365=as.numeric(c10))  
    tree_model<-readRDS("tree_model_tuned.rds")
    tree_pred<-predict(tree_model,df)
    return(tree_pred)
}


rf_prediction<- function(c1,c2,c3,c4,c5,c6,c7,c8,c9,c10)
{
    df<- data.frame(neighbourhood_group=as.factor(c1),neighbourhood=as.factor(c2),latitude=as.numeric(c3),longitude=as.numeric(c4),room_type=as.factor(c5),minimum_nights=as.numeric(c6),number_of_reviews=as.numeric(c7),reviews_per_month=as.numeric(c8),calculated_host_listings_count=as.numeric(c9),availability_365=as.numeric(c10))  
    rf_model<-readRDS("rf_model_tuned.rds")
    rf_pred<-predict(rf_model,df)
    return(rf_pred)
}


# Define server logic required to draw a histogram
server <- function(input, output) {
    
    observeEvent(input$model_selector,{
        model<- input$model_selector
        ne_group<- input$neighbourhood_group
        ne<- input$neighbourhood
        lat<-input$lat
        long<-input$long
        room<- input$room_type
        nights<- input$nights
        reviews<- input$reviews
        reviews_pm<- input$reviews_pm
        host<- input$host
        avail<- input$available
        if(model=="Support Vector Machine")
        {
            output$model<-renderText("<b>Making predictions using SVM</b>")
            out<-svm_prediction(ne_group,ne,lat,long,room,nights,reviews,reviews_pm,host,avail)
            #output$negr<-renderText(ne_group)
            output$pred<-renderPrint(out)
            #output$df<-renderPrint(final_df)
        }
        if(model=="Decision Tree Classifier")
        {
            output$model<-renderText("<b>Using Decision Trees to make predictions</b>")
            out<- tree_prediction(ne_group,ne,lat,long,room,nights,reviews,reviews_pm,host,avail)
            output$pred<-renderPrint(out)
        }
        
        if(model=="Random Forest Classifier")
        {
            output$model<-renderText("<b>Using Random Forest Classifier to make predictions</b>")
            out<- rf_prediction(ne_group,ne,lat,long,room,nights,reviews,reviews_pm,host,avail)
            output$pred<-renderPrint(out)
        }
        
    })
    
}

# Run the application 
shinyApp(ui = ui, server = server)
