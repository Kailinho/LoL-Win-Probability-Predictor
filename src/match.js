const axios = require('axios');

const riotApiBaseUrl = 'https://americas.api.riotgames.com/';
const apiKey = 'RGAPI-d5f34a87-0ed4-400a-9481-893e13ea4e6e'; // Replace with your actual API key
const headers = {
    'X-Riot-Token': apiKey,
};
const summonerName = 'Kailinho';
const puuid = `-23srJpsUju9lhDerocvX4vyCUBo1hy_7vmUK-4Lrd8SWwvQhKNXXrN59DzJCfe7fDuwwJ96HoPx-A`;
const matchListUrl = `${riotApiBaseUrl}lol/match/v5/matches/by-puuid/${puuid}/ids?start=1&count=1&api_key=${apiKey}`;

// function to get the list of matches
async function getMatches() {
    try {
        const response = await axios.get(matchListUrl, { headers });
        const matchIdList = response.data;

        console.log('Match IDs:', matchIdList);


        for (const matchId of matchIdList) {
            await getMatchInfo(matchId);
        }
    } catch (error) {
        console.error(`Error: ${error.response?.status}, ${error.response?.data}`);
    }
}


// function to get the match details
async function getMatchInfo(matchId) {
    console.log(`Fetching match details for match ${matchId}...`);
    try {
        const matchTimelineUrl = `${riotApiBaseUrl}lol/match/v5/matches/${matchId}/timeline?api_key=${apiKey}`;
        const matchTimelineResponse = await axios.get(matchTimelineUrl);
        const matchParticipants = matchTimelineResponse.data.metadata.participants;
        const matchTimeline = matchTimelineResponse.data.info.frames;
        const aggregatedData = collectAggregatedData(matchTimeline);

    } catch (error) {
        if (error.response) {
            console.error('Error:', error.response.status, error.response.data);
        } else {
            console.error('Error making the request:', error.message);
        }
    }
}


async function collectAggregatedData(frames) {

    keylist = ['assists','champLevel','damageDealtToObjectives','damageDealtToTurrets','damageSelfMitigated','deaths','doubleKills','goldEarned','goldSpent','inhibitorKills','killingSprees','kills','largestCriticalStrike','largestKillingSpree','largestMultiKill','longestTimeSpentLiving','magicDamageDealt','magicDamageDealtToChampions','magicalDamageTaken','neutralMinionsKilled','pentaKills','physicalDamageDealt','physicalDamageDealtToChampions','physicalDamageTaken','quadraKills','sightWardsBoughtInGame','timeCCingOthers','totalDamageDealt','totalDamageDealtToChampions','totalDamageTaken','totalHeal','totalMinionsKilled','totalTimeCrowdControlDealt','totalUnitsHealed','tripleKills','trueDamageDealt','trueDamageDealtToChampions','trueDamageTaken','turretKills','visionScore','visionWardsBoughtInGame','win']
    try{
        const kills = {};
        const deaths = {};
        const assists = {};
        i=0;
        frames.forEach(frame => {

            const participantFrame = frame.participantFrames;
            const eventFrame = frame.events;
            const combinedData = {
                participantFrame,
                events: eventFrame,
            };
               
                    // Iterate through events to update kill/death/assist counts
            combinedData.events.forEach(event => {
                if (event.type === "CHAMPION_KILL") {
                    // Increment kill count for killer
                    if (!kills[event.killerId]) {
                        kills[event.killerId] = 1;
                    } else {
                        kills[event.killerId]++;
                    }
    
                    // Increment death count for victim
                    if (!deaths[event.victimId]) {
                        deaths[event.victimId] = 1;
                    } else {
                        deaths[event.victimId]++;
                    }
    
                    // Increment assist count for assisting participants
                    if (event.assistingParticipantIds) {
                        event.assistingParticipantIds.forEach(assistId => {
                            if (!assists[assistId]) {
                                assists[assistId] = 1;
                            } else {
                                assists[assistId]++;
                            }
                        });
                    }
                }
            });
            console.log('Kills:', kills);
            console.log('Deaths:', deaths);
            console.log('Assists:', assists);
        });    
    }catch(error){
        console.error('Error:', error.message);
    }

    


    // for (const frame of frames) {
    //     const participantFrames = frame.participantFrames;

    //     for (const participantId in participantFrames) {
    //         if (participantFrames.hasOwnProperty(participantId)) {
    //             const participantFrame = participantFrames[participantId];

    //             const currentGold = participantFrame.currentGold;
    //             const level = participantFrame.level;
    //             const minionsKilled = participantFrame.minionsKilled;

    //             allCurrentGold.push(currentGold);
    //             allLevel.push(level);
    //             allMinionsKilled.push(minionsKilled);
    //         }
    //     }
    // }

    // return {
    //     allCurrentGold,
    //     allLevel,
    //     allMinionsKilled,
    // };
}


getMatches();
