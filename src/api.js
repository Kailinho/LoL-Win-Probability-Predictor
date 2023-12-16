const axios = require('axios');


const apiKey = 'RGAPI-3fb1dc4d-f2e3-4fd4-98fe-962d1739bec6';
const riotIDs = ['Kailinho/NA1','Sophist Sage/0409','Skollie/420','Isles2/NA1','Refugy/NA1','DuoSumo/NA1'];  
const puuids = [];
const headers = {
    'X-Riot-Token': apiKey,
};


const axiosPromises = riotIDs.map(summonerName => {
    const url = `https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/${summonerName}?api_key=${apiKey}`;
    return axios.get(url, { headers })
        .then(response => {
            const summonerData = response.data;
            const puuid = summonerData.puuid;
            puuids.push(puuid);
            console.log(`puuid: ${puuid}`);
            console.log(`Summoner ID: ${summonerData.gameName + summonerData.tagLine}`);
        })
        .catch(error => {
            console.log(error);
        });
});

Promise.all(axiosPromises)
    .then(() => {
        console.log('puuids:', puuids);
    })
    .catch(error => {
        console.log('Error:', error);
    });