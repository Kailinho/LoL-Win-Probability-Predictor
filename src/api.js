const axios = require('axios');


const apiKey = 'RGAPI-7c36b354-8d93-429e-a849-9b225e83eb54';
const riotIDs = ['Yozu/Lux','Unyielding/NA1','Skollie/420','aishieryu/alive','Anyday/1221','Solaros/NA1','ToasyAlex/NA1','Legacy/L77','DrunkCatalyst/NA1'];  
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