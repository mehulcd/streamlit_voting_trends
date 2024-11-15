import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet
#import time_series_model

# Load your dataset
data = pd.read_csv("congress_voting_data_flattened.csv")
data['bill_introduced_date'] = pd.to_datetime(data['bill_introduced_date'], utc=True, format = 'mixed').dt.tz_localize(None)
#df['bill_introduced_date'].dt.to_period('M')
data['month'] = data['bill_introduced_date'].dt.to_period('M')

data_senate = data[data['chamber'] == 'Senate']

# Create the state-level summary data
df = data_senate.copy()
state_data = {}
for column in df.columns:
    if 'votes_yea_' in column or 'votes_nay_' in column:

        if 'bill_votes_yea' in column or 'bill_votes_nay' in column:
            parts = column.split('_')
            vote_type = parts[2]
            state_abbr = parts[3]

        else:

            parts = column.split('_')
            vote_type = parts[1]
            state_abbr = parts[2]

        total_votes = df[column].sum()

        if state_abbr not in state_data:
            state_data[state_abbr] = {'yea_votes': 0, 'nay_votes': 0, 'democrat_yea': 0, 'republican_yea': 0, 
                                      'democrat_nay': 0, 'republican_nay': 0}

        if vote_type == 'yea':
            state_data[state_abbr]['yea_votes'] += total_votes
            # Assuming columns have party information for further breakdown, e.g., 'votes_yea_democrat_TX'
            if 'democrat' in column:
                state_data[state_abbr]['democrat_yea'] += total_votes
            elif 'republican' in column:
                state_data[state_abbr]['republican_yea'] += total_votes
        elif vote_type == 'nay':
            state_data[state_abbr]['nay_votes'] += total_votes
            if 'democrat' in column:
                state_data[state_abbr]['democrat_nay'] += total_votes
            elif 'republican' in column:
                state_data[state_abbr]['republican_nay'] += total_votes

# Convert state_data to DataFrame
state_df = pd.DataFrame(state_data).T.reset_index()
state_df.columns = ['state', 'yea_votes', 'nay_votes', 'democrat_yea', 'republican_yea', 'democrat_nay', 'republican_nay']


state_df['yea_minus_nay'] = state_df['yea_votes'] - state_df['nay_votes']
state_df['for_legislation_percent'] = (state_df['yea_votes']/(state_df['yea_votes']+state_df['nay_votes'])) * 100
# convert nan values to 0
state_df.fillna(0, inplace=True)

sttate_df_wo_total = state_df[state_df['state'] != 'total']

# Plotting the choropleth map with enhanced interactivity and tooltips using Plotly
fig = px.choropleth(
    sttate_df_wo_total,
    locations='state',
    locationmode='USA-states',
    color='yea_minus_nay',
    scope='usa',
    title='Senate Voting Trend (Yea vs. Nay) by State',
    labels={'yea_minus_nay': 'Yea - Nay Votes'},
    color_continuous_scale='RdBu',
    #range_color=[-100, 100],
    hover_name='state',
    template='plotly_dark',
    #width=800,  # Set the width of the map (in pixels)
    hover_data={
        #'yea_votes': True,
        #'nay_votes': True,
        'for_legislation_percent': True,
    }
)

# Increase map size
fig.update_layout(
    width=800,  # Set the width of the map (in pixels)
    height=600,  # Set the height of the map (in pixels)
)
# Streamlit App for Interactivity
st.title('Interactive Senate Voting Trend (Yea vs. Nay) by State')

# Display the choropleth map
st.plotly_chart(fig)


# Party Vote Share for Each State:
# Aggregation: Calculate the proportion of votes for each party (Democrat and Republican) across all sessions for each state.
# Visualization: Use a color scale to represent the share of votes for each party. A state with a larger proportion of Democratic votes could be colored one way, and a larger proportion of Republican votes another.
# Calculate the proportion of votes for each party (Democrat and Republican) across all sessions for each state
state_df['democrat_vote_share'] = (state_df['democrat_yea'] / (state_df['democrat_yea'] + state_df['democrat_nay'])) * 100
state_df['republican_vote_share'] = (state_df['republican_yea'] / (state_df['republican_yea'] + state_df['republican_nay'])) * 100
# convert nan values to 0
state_df.fillna(0, inplace=True)

#print (state_df)

# Add a new column to determine the color based on which party has a higher vote share
state_df['color'] = state_df.apply(
    lambda row: 'red' if row['republican_vote_share'] > row['democrat_vote_share'] else 'blue',
    axis=1
)


# Plotting the choropleth map showing Red for Republican and Blue for Democrat based on which party has a higher vote share
fig_party_vote_share = px.choropleth(
    state_df,
    locations='state',
    locationmode='USA-states',
    # color should be based on the party with the higher vote share
    color='color',
    scope='usa',
    title='Senate : Showing Party Vote Share by State',
    labels={'democrat_vote_share': 'Democrat Vote Share (%)', 'republican_vote_share': 'Republican Vote Share (%)'},
    #color_continuous_scale='Blues',
    color_discrete_map={'red': 'red', 'blue': 'blue'},
    hover_name='state',
    template='plotly_dark',
    hover_data={
        'democrat_vote_share': True,
        'republican_vote_share': True
    },
    width=800,  # Set the width of the map (in pixels)
)

# Increase map size
fig_party_vote_share.update_layout(
    width=800,  # Set the width of the map (in pixels)
    height=600,  # Set the height of the map (in pixels)
)

# Display the choropleth map
st.plotly_chart(fig_party_vote_share)


def forecast (monthly_data, dependent_variable):

    monthly_data['ds'] = monthly_data['month'].dt.to_timestamp()
    monthly_data['y'] = monthly_data[dependent_variable]

    model = Prophet(interval_width=0.95, weekly_seasonality=True, daily_seasonality=True)
    #model = Prophet(interval_width=0.95)
    model.fit(monthly_data)

    # Step 6: Create a future dataframe and make predictions
    future = model.make_future_dataframe(periods=48, freq='M')  # Forecasting 12 months into the future
    forecast = model.predict(future)

    # smoothing the forecasted values
    forecast['yhat'] = forecast['yhat'].rolling(window=3).mean()

    # Display forecasted values
    #print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    return forecast

def display_state_voting_trend(clicked_state):
    
    if clicked_state == 'total' :

        clicked_state = clicked_state.lower()
        columns_to_check = [
            f'bill_votes_yea_{clicked_state}_democrat',
            f'bill_votes_nay_{clicked_state}_democrat',
            f'bill_votes_yea_{clicked_state}_republican',
            f'bill_votes_nay_{clicked_state}_republican'
        ]

    else :
        columns_to_check = [
            f'votes_yea_{clicked_state}_democrat',
            f'votes_nay_{clicked_state}_democrat',
            f'votes_yea_{clicked_state}_republican',
            f'votes_nay_{clicked_state}_republican'
        ]

    data_senate_state = data_senate[['month'] + columns_to_check]

    # Display the time series line chart
    monthly_data = data_senate_state.groupby('month').sum().reset_index()
    monthly_data['yea_minus_nay_democrat'] = monthly_data [columns_to_check[0]] - monthly_data[columns_to_check[1]]
    monthly_data['yea_minus_nay_republican'] = monthly_data[columns_to_check[2]] - monthly_data[columns_to_check[3]]

    # Display the time series line chart for yea - nay votes for the selected state fro both parties
    # Convert 'month' column to string
    monthly_data['month_str'] = monthly_data['month'].astype(str)
    fig_time_series = px.line(
        monthly_data,
        x='month_str',
        y=['yea_minus_nay_democrat', 'yea_minus_nay_republican'],
        title=f'Senate Voting Trend (Yea vs. Nay) in {clicked_state} by Party',
        labels={'value': 'Yea - Nay Votes Difference', 'variable': 'Party'},
        line_shape='linear',
        #line_shape_sequence=['linear', 'spline', 'vhv', 'hvh', 'vh', 'hv'],
        template='plotly_dark'
    )

    # increase the size of the chart
    fig_time_series.update_layout(width=800, height=500)
    fig_time_series.update_xaxes(title_text='Month')

    st.plotly_chart(fig_time_series)

    # Do a time series forecast using the time_series_model.py script
    forecast_senate_dem = forecast(monthly_data, 'yea_minus_nay_democrat')
    forecast_senate_rep = forecast(monthly_data, 'yea_minus_nay_republican')

    # Filter forecasted values from 2024 to 2025
    forecast_senate_dem = forecast_senate_dem[forecast_senate_dem['ds'].dt.year >= 2024]
    forecast_senate_rep = forecast_senate_rep[forecast_senate_rep['ds'].dt.year >= 2024]


    # Plot the forecasted values using Plotly line chart
    fig_forecast = px.line(
        forecast_senate_dem,
        x='ds',
        y=['yhat'],
        title=f'Forecasted Senate Voting Trend (Yea vs. Nay) in {clicked_state} for Democrat Party',
        labels={'value': 'Yea - Nay Votes Difference', 'variable': 'Party'},
        line_shape='spline'
    )

    fig_forecast.update_xaxes(title_text='Month')
    st.plotly_chart(fig_forecast)

    # Plot the forecasted values using Plotly line chart
    fig_forecast = px.line(
        forecast_senate_rep,
        x='ds',
        y=['yhat'],
        title=f'Forecasted Senate Voting Trend (Yea vs. Nay) in {clicked_state} for Republican Party',
        labels={'value': 'Yea - Nay Votes Difference', 'variable': 'Party'},
        line_shape='spline'
    )

    fig_forecast.update_xaxes(title_text='Month')
    st.plotly_chart(fig_forecast)

    # Display bar chart for party-wise votes
    if clicked_state == 'total' :
        details = state_df[state_df['state'] == 'total'].iloc[0]
    else:
        details = state_df[state_df['state'] == clicked_state].iloc[0]
    
    party_votes = pd.DataFrame({
        'Party': ['Democrat', 'Republican'],
        'Yea Votes': [details['democrat_yea'], details['republican_yea']],
        'Nay Votes': [details['democrat_nay'], details['republican_nay']]
    })

    fig_party_votes = px.bar(
        party_votes,
        x='Party',
        y=['Yea Votes', 'Nay Votes'],
        title=f'Party-wise Votes in {clicked_state}',
        labels={'value': 'Votes', 'variable': 'Vote Type'},
        barmode='group',
        template='plotly_dark'
    )

    fig_party_votes.update_xaxes(title_text='Party')
    st.plotly_chart(fig_party_votes)


# Ensure state_dr['state'] does not contain 'yea', 'nay', 'total' values
state_df = state_df[~state_df['state'].str.contains('yea|nay', case=False)]

# Get the state clicked by the user
state_list = state_df['state'].unique()
# convert state_list to upper case
state_list = [state.upper() for state in state_list]
click_data = st.selectbox('Select a State', state_list)

if click_data:

    if click_data == 'TOTAL':
        selected_state = 'total'
    else:
        selected_state = click_data
    st.write(f"Selected State: {selected_state}")
    display_state_voting_trend(selected_state)