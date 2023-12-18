const axios = require('axios');


const apiKey = 'RGAPI-8ea7f2b5-052f-4a16-9b0b-f97786be472b';
const riotIDs = ['Yozu/Lux','Unyielding/NA1','Skollie/420','aishieryu/alive','Anyday/1221','Solaros/NA1'];  
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