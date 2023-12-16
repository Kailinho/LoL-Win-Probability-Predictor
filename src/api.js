const axios = require('axios');

// Replace 'YOUR_API_KEY' with your actual Riot Games API key
const apiKey = 'RGAPI-d5f34a87-0ed4-400a-9481-893e13ea4e6e';
const summonerName = 'Kailinho';  // Replace with the summoner name you want to query

// Define the API endpoint for summoner information
const url = `https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/${summonerName}`;

// Set up headers with your API key
const headers = {
    'X-Riot-Token': apiKey,
};

// Make the API request
axios.get(url, { headers })
    .then(response => {
        // Access data from the response
        const summonerData = response.data;

        // Display some information about the summoner
        console.log(`Summoner Name: ${summonerData.name}`);
        console.log(`Summoner Level: ${summonerData.summonerLevel}`);
        console.log(`Summoner ID: ${summonerData.id}`);
        console.log(`puuid: ${summonerData.puuid}`);
    })
    .catch(error => {
        // Handle errors
        console.error(`Error: ${error.response.status}, ${error.response.data}`);
    });
