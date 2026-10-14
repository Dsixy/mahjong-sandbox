const $ = (id) => document.getElementById(id);
const SUITS = ['万', '筒', '条'];
const NUMBERS = '一二三四五六七八九';
const PHASES = {exchange: '换三张', declare_suit: '定缺', self_turn: '行牌', discard_response: '行牌', rob_kong_response: '抢杠', terminal: '结算'};
const REASONS = {ron: '荣和', tsumo: '自摸', concealed_kong: '暗杠', open_kong: '大明杠', added_kong: '加杠', flower_pig: '查花猪', noten: '查大叫', kong_refund: '退杠分', kong_discard_transfer: '呼叫转移', riichi_deposit:'立直棒', riichi_award:'立直棒奖励', honba:'本场', nagashi_mangan:'流局满贯', pao:'包牌'};
const POLICY_NAMES = {greedy: '牌效基线 AI', 'win-first': '胡牌优先 AI', random: '随机 AI'};
let view = null, liveView = null, selected = [], selectedSuit = null;
let busy = false, generation = 0, replayRequest = 0, replayTimer = null, playingReplay = false;
let animations = localStorage.getItem('mahjong-animation') !== 'off';
let sound = false, audio = null, toastTimer = null;
const announcedResults = new Set();
const HONORS = ['东','南','西','北','白','发','中'];
const tileName = (tile) => Math.floor(tile/4)>=27 ? HONORS[Math.floor(tile/4)-27] : `${view?.observation.red_fives?.includes(tile) ? '赤' : ''}${NUMBERS[Math.floor(tile / 4) % 9]}${SUITS[Math.floor(tile / 36)]}`;
const END_NAMES={three_winners:'三家已胡，血战结束。', wall_exhausted:'荒牌流局', ron:'荣和，本局结束。', tsumo:'自摸，本局结束。', nine_terminals:'九种九牌', four_winds:'四风连打', four_riichi:'四家立直', four_kans:'四杠散了', triple_ron:'三家和了'};
const reasonName = reason => view?.observation.variant==='riichi' && reason==='noten' ? '听牌罚符' : (REASONS[reason] || reason);
const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4000);
}

async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? {Accept:'application/json'} : {'Content-Type':'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '牌桌暂时不可用');
    $('connection').classList.remove('offline');
    return data;
  } catch (error) {
    if (error instanceof TypeError || error.name === 'AbortError') {
      $('connection').classList.add('offline');
      throw new Error('连接中断，请确认本地牌桌服务仍在运行后重试。');
    }
    throw error;
  } finally { clearTimeout(timeout); }
}

function tone(type = 'tile') {
  if (!sound || !audio) return;
  const oscillator = audio.createOscillator(), gain = audio.createGain();
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(type === 'win' ? 760 : 480, audio.currentTime);
  oscillator.frequency.exponentialRampToValueAtTime(type === 'win' ? 1100 : 220, audio.currentTime + .06);
  gain.gain.setValueAtTime(.055, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(.001, audio.currentTime + .09);
  oscillator.connect(gain); gain.connect(audio.destination);
  oscillator.start(); oscillator.stop(audio.currentTime + .1);
}

function tileElement(tile, {back=false, button=false, selectedTile=false, disabled=false, drawn=false, called=false, win=false, index=0}={}) {
  const node = document.createElement(button ? 'button' : 'span');
  node.className = ['tile', selectedTile && 'selected', drawn && 'is-drawn', called && 'called', win && 'winning-discard'].filter(Boolean).join(' ');
  const image = document.createElement('img');
  image.src = `/assets/tiles/${back ? 'back' : Math.floor(tile / 4)+(view?.observation.red_fives?.includes(tile)?'-red':'')}.svg`;
  image.alt = back ? '暗牌' : tileName(tile);
  image.draggable = false;
  node.append(image);
  if (button) {
    node.type = 'button'; node.disabled = disabled;
    node.dataset.tileId = tile;
    node.setAttribute('aria-label', `${tileName(tile)}，手牌第 ${index + 1} 张`);
    node.setAttribute('aria-pressed', String(selectedTile));
    node.addEventListener('click', () => chooseTile(tile));
    node.addEventListener('dblclick', () => {
      if (busy || view?.mode !== 'play' || view.observation.phase !== 'self_turn') return;
      const action = view.observation.legal_actions.find((a) => a.kind === 'discard' && a.tiles[0] === tile);
      if (action) submitAction(action);
    });
  }
  return node;
}

function meldElement(meld) {
  const node = document.createElement('div'); node.className = 'meld';
  meld.tiles.forEach((tile) => node.append(tileElement(tile)));
  const label = document.createElement('span'); label.className = 'meld-label';
  label.textContent = {chi:'吃', peng:'碰', open_kong:'杠', concealed_kong:'暗杠', added_kong:'加杠'}[meld.kind];
  node.title = `${label.textContent} ${tileName(meld.tiles[0])}`; node.append(label);
  return node;
}

function profileMarkup(player, seat) {
  const own = seat === 0, win = view.observation.wins.find((w) => w.player === seat);
  const active = view.observation.actor === seat && view.observation.phase !== 'terminal';
  const tag = player.riichi ? '<span class="player-tag won">立直</span>' : win ? '<span class="player-tag won">已胡</span>' : player.missing_suit !== null ? `<span class="player-tag">缺${SUITS[player.missing_suit]}</span>` : '';
  const dealer = view.observation.events.find((e) => e.kind === 'round_started')?.data.dealer === seat ? '<span class="player-tag dealer">庄</span>' : '';
  const score = (view.match?.final_scores || view.observation.scores)[seat];
  return `<div class="player-profile ${active ? 'active' : ''}"><div class="avatar-wrap"><img src="/assets/avatar-${seat}.svg" alt="${escape(view.names[seat])}的头像"></div><div><div class="player-name">${escape(view.names[seat])}${dealer}${player.seat_wind !== undefined ? `<span class="player-tag">${HONORS[player.seat_wind-27]}</span>` : ''}${tag}${player.furiten ? '<span class="player-tag">振听</span>' : ''}</div><div class="player-score">${score > 0 ? '+' : ''}${score} 分${own ? '' : ` · ${player.concealed_count} 张`}</div></div></div>`;
}

function eventText(event) {
  const data = event.data, name = (seat) => seat===4 ? '供托' : view.names[seat];
  switch (event.kind) {
    case 'round_started': return '四方入座，新局开始。';
    case 'dealt': return data.player === 0 ? '发牌完成，看看你的手气。' : null;
    case 'exchange_selected': return `你换出 ${data.tiles.map(tileName).join('、')}`;
    case 'exchange_completed': return `换三张完成，换给${['', '下家', '对家', '上家'][data.offset]}。${data.received ? `收到 ${data.received.map(tileName).join('、')}` : ''}`;
    case 'missing_suits_declared': return data.suits.map((suit, seat) => `${name(seat)}缺${SUITS[suit]}`).join(' · ');
    case 'draw': return `${name(data.player)}${data.source === 'kong' ? '杠后补牌' : '摸牌'}${data.tile === undefined ? '' : `：${tileName(data.tile)}`}`;
    case 'discard': return `${name(data.player)}打出${tileName(data.tile)}${data.tsumogiri ? '（摸切）' : ''}`;
    case 'chi': return `${name(data.player)}吃 ${data.meld.tiles.map(tileName).join('、')}`;
    case 'riichi_declared': return `${name(data.player)}宣言立直`;
    case 'riichi_accepted': return `${name(data.player)}${data.double ? '两立直' : '立直'}成立，存入 1000 点`;
    case 'dora_revealed': return `宝牌指示牌：${data.indicators.map(tileName).join('、')}`;
    case 'ura_revealed': return `里宝牌指示牌：${data.indicators.map(tileName).join('、')}`;
    case 'peng': return `${name(data.player)}碰${tileName(data.meld.tiles[0])}`;
    case 'kong_proposed': return `${name(data.player)}准备${REASONS[data.kind] || '加杠'}${tileName(data.tile)}`;
    case 'kong_confirmed': return `${name(data.player)}${REASONS[data.meld.kind]}${tileName(data.meld.tiles[0])}`;
    case 'kong_robbed': return `${name(data.player)}的加杠被抢`;
    case 'win': return `${name(data.win.player)}${data.win.self_draw ? '自摸' : '荣和'}！${data.win.value.fan} 番`;
    case 'payment': return `${name(data.payer)} → ${name(data.receiver)} · ${data.amount} 分（${reasonName(data.reason)}）`;
    case 'round_ended': return END_NAMES[data.reason] || data.reason;
    default: return null;
  }
}

function render(nextView) {
  const oldDecision = view?.observation.decision_id, oldMode = view?.mode;
  view = nextView;
  if (oldDecision !== view.observation.decision_id || oldMode !== view.mode) { selected = []; selectedSuit = null; }
  const obs = view.observation, phase = obs.phase, replayMode = view.mode === 'replay';
  document.body.classList.toggle('busy', busy);
  $('nav-play').classList.toggle('active', !replayMode); $('nav-replay').classList.toggle('active', replayMode);
  $('policy-badge').textContent = replayMode ? '已校验牌谱' : POLICY_NAMES[view.settings.policy];
  $('tiles-left').textContent = obs.remaining_tiles;
  $('center-phase').textContent = PHASES[phase];
  const riichi=obs.variant==='riichi';
  $('variant-eyebrow').textContent=riichi?'RIICHI MAHJONG':'SICHUAN MAHJONG';
  $('variant-title').textContent=riichi?'四人立直':'血战到底';
  $('variant-intro').textContent=riichi?'一番一符，四方过招。':'一桌川麻，四方过招。';
  $('variant-help').innerHTML=riichi?'立直、振听与宝牌。<br>和牌或流局后，本局结束。':'胡牌后，其余玩家继续行牌。<br>三家胡牌或牌尽，本局结束。';
  $('sichuan-rules').hidden=riichi; $('riichi-rules').hidden=!riichi;
  $('riichi-info').hidden=!riichi;
  if(riichi){
    $('hand-name').textContent=`${HONORS[obs.round_wind-27]}${obs.hand_number}局 · ${obs.honba}本场`;
    $('riichi-bank').textContent=`供托 ${obs.riichi_sticks} 棒`;
    $('dora-tiles').replaceChildren(); obs.dora_indicators.forEach(t=>$('dora-tiles').append(tileElement(t)));
  }
  $('progress-list').children[2].querySelector('small').textContent=riichi?'吃碰杠，立直与和牌':'碰杠胡，血战到底';
  [...$('progress-list').children].slice(0,2).forEach(li=>li.hidden=riichi);
  $('active-players').textContent = `${obs.players.filter((p) => p.can_act).length} 家行牌`;
  $('round-status').textContent = replayMode ? '复盘中' : PHASES[phase];
  const stage = phase === 'exchange' ? 0 : phase === 'declare_suit' ? 1 : phase === 'terminal' ? 3 : 2;
  [...$('progress-list').children].forEach((li, index) => { li.classList.toggle('current', index === stage); li.classList.toggle('done', index < stage); });
  obs.players.forEach((player, seat) => {
    const node = $(`player-${seat}`); node.innerHTML = profileMarkup(player, seat);
    if (!seat) return;
    node.classList.toggle('won-player', !player.can_act);
    const cards = document.createElement('div'); cards.className = 'opponent-hand';
    const tiles = player.concealed ? [...player.concealed] : Array(player.concealed_count).fill(null);
    const win = obs.wins.find((w) => w.player === seat && !w.self_draw);
    if (player.concealed && win) tiles.push(win.tile);
    tiles.forEach((tile) => cards.append(tileElement(tile, {back:tile === null})));
    node.append(cards);
    const melds = document.createElement('div'); melds.className = 'opponent-melds';
    player.melds.forEach((meld) => melds.append(meldElement(meld))); node.append(melds);
  });
  for (let seat = 0; seat < 4; seat++) {
    const river = $(`river-${seat}`); river.replaceChildren();
    const caption = document.createElement('span'); caption.className = 'river-caption'; caption.textContent = `${view.names[seat]}的牌河`;
    river.append(caption);
    if (!obs.players[seat].river.length) {
      const empty = document.createElement('span'); empty.className = 'empty-river'; empty.textContent = '尚未出牌'; river.append(empty);
    } else {
      const grid = document.createElement('span'); grid.className = 'river-grid';
      obs.players[seat].river.forEach((discard,index) => {const tile=tileElement(discard.tile, {called:discard.called_by !== null, win:discard.winner_refs.length > 0}); if(index===obs.players[seat].riichi_discard)tile.classList.add('riichi-discard'); grid.append(tile);});
      river.append(grid);
    }
  }
  const activity = obs.events.map((event) => ({event, text:eventText(event)})).filter((item) => item.text).reverse();
  $('activity').replaceChildren();
  activity.slice(0, 28).forEach((item) => { const node = document.createElement('div'); node.className = 'activity-entry'; node.textContent = item.text; $('activity').append(node); });
  const recent = activity.find((item) => item.event.kind !== 'payment');
  $('table-notice').textContent = phase === 'exchange' ? '各选同门三张，所有玩家选好后一起交换。' : phase === 'declare_suit' ? '定缺同时公开，先选择你的缺门。' : recent?.text || '';
  renderHand(); renderActions();
  $('replay-current').disabled = busy || !liveView?.observation.end_reason;
  $('new-game').disabled = busy;
  $('nav-replay').disabled = busy;
  $('footer-caption').textContent = replayMode ? '牌谱回放 · 可逐步查看 · 全桌视角仅用于复盘' : riichi ? '四人立直 · 一番缚 · 番符计分' : '四川麻将 · 定缺 · 换三张 · 血战到底';
  $('replay-controls').hidden = !replayMode;
  if (replayMode) renderReplayControls();
}

function renderHand() {
  const obs = view.observation, own = obs.players[0], playable = view.mode === 'play';
  let tiles = [...own.concealed];
  if (own.last_drawn !== null && obs.phase === 'self_turn' && tiles.includes(own.last_drawn)) tiles = tiles.filter((tile) => tile !== own.last_drawn).concat(own.last_drawn);
  const win = obs.wins.find((w) => w.player === 0 && !w.self_draw);
  if (win) tiles.push(win.tile);
  $('own-hand').replaceChildren();
  tiles.forEach((tile, index) => {
    let eligible = false;
    if (playable && obs.phase === 'exchange') eligible = obs.legal_actions.some((a) => a.kind === 'exchange' && a.tiles.includes(tile));
    if (playable && obs.phase === 'self_turn') eligible = obs.legal_actions.some((a) => a.kind === 'discard' && a.tiles[0] === tile);
    const otherSuit = obs.phase === 'exchange' && selected.length && Math.floor(selected[0] / 36) !== Math.floor(tile / 36);
    const node = tileElement(tile, {button:playable && own.can_act, disabled:busy || !eligible || !!otherSuit,
      selectedTile:selected.includes(tile), drawn:obs.phase === 'self_turn' && tile === own.last_drawn,
      win:!!win && tile === win.tile, index});
    if (playable && own.can_act && (!eligible || otherSuit) && ['exchange','self_turn'].includes(obs.phase)) node.classList.add('dimmed');
    $('own-hand').append(node);
  });
  $('own-melds').replaceChildren(); own.melds.forEach((meld) => $('own-melds').append(meldElement(meld)));
  let instruction = '等待其他玩家落牌…';
  if (view.mode === 'replay') instruction = '复盘中的手牌，可逐步查看变化';
  else if (obs.phase === 'terminal') instruction = '本局已结束，可以查看结算与复盘';
  else if (!own.can_act) instruction = '你已胡牌，观看其余玩家继续行牌';
  else if (obs.phase === 'exchange') instruction = `选择同一门的三张牌 · 已选 ${selected.length} / 3`;
  else if (obs.phase === 'declare_suit') instruction = '选择本局缺门，有缺门牌时先打出';
  else if (obs.legal_actions.some((a) => a.kind === 'discard')) instruction = own.missing_suit !== null && own.concealed.some((tile) => Math.floor(tile / 36) === own.missing_suit) ? `请先打出${SUITS[own.missing_suit]}牌` : '轮到你了，选一张牌打出';
  else if (obs.legal_actions.length) instruction = obs.pending_tile !== null ? `${view.names[obs.actor]}的${tileName(obs.pending_tile)}，你要应牌吗？` : '选择你的操作';
  $('hand-instruction').textContent = instruction;
  $('hand-hint').textContent = view.mode === 'replay' ? '逐步查看牌局' : obs.phase === 'terminal' ? '查看结算 · 保存牌谱' : obs.phase === 'exchange' ? '同一花色 · 三张一组' : '点击选牌 · 双击出牌';
}

function chooseTile(tile) {
  if (busy || view.mode !== 'play') return;
  if (view.observation.phase === 'exchange') {
    if (selected.includes(tile)) selected = selected.filter((id) => id !== tile);
    else if (selected.length < 3 && (!selected.length || Math.floor(selected[0] / 36) === Math.floor(tile / 36))) selected.push(tile);
    else if (selected.length === 3) { toast('已经选好三张，取消一张后可重新选择。'); return; }
  } else selected = selected[0] === tile ? [] : [tile];
  tone(); renderHand(); renderActions();
}

function actionButton(label, callback, {primary=false, gold=false, disabled=false, className=''}={}) {
  const button = document.createElement('button'); button.type = 'button';
  button.className = `button ${primary ? 'primary' : gold ? 'gold' : 'outline'} ${className}`;
  button.textContent = label; button.disabled = busy || disabled; button.addEventListener('click', callback);
  $('actions').append(button); return button;
}

function renderActions() {
  const obs = view.observation; $('actions').replaceChildren();
  if (view.mode === 'replay') { const label = document.createElement('span'); label.className = 'action-description'; label.textContent = '回放模式 · 使用下方进度条查看每一步'; $('actions').append(label); if(obs.phase==='terminal')actionButton('查看结算',showSettlement,{primary:true}); return; }
  if (obs.phase === 'terminal') { actionButton('查看结算', showSettlement, {primary:true}); if(view.match && !view.match.ended)actionButton('下一局', nextHand, {primary:true}); else actionButton('再来一局', () => $('new-dialog').showModal()); return; }
  if (obs.phase === 'exchange') {
    const ids = [...selected].sort((a,b) => a-b);
    const action = obs.legal_actions.find((a) => a.kind === 'exchange' && a.tiles.length === ids.length && a.tiles.every((id, i) => id === ids[i]));
    actionButton(`确认换牌 · ${selected.length}/3`, () => submitAction(action), {primary:true, disabled:!action});
    actionButton('重新选择', () => { selected=[]; renderHand(); renderActions(); }, {disabled:!selected.length}); return;
  }
  if (obs.phase === 'declare_suit') {
    SUITS.forEach((name, suit) => {
      const button = document.createElement('button'); button.type='button'; button.className=`suit-option ${selectedSuit === suit ? 'selected' : ''}`;
      const count = obs.players[0].concealed.filter((tile) => Math.floor(tile / 36) === suit).length;
      button.innerHTML=`<b class="suit-glyph">${name}</b><span>缺${name}<small>手中 ${count} 张</small></span>`;
      button.disabled=busy; button.setAttribute('aria-pressed', String(selectedSuit === suit));
      button.addEventListener('click', () => { selectedSuit=suit; renderActions(); }); $('actions').append(button);
    });
    actionButton(selectedSuit === null ? '确认定缺' : `确认缺${SUITS[selectedSuit]}`, () => submitAction(obs.legal_actions.find((a) => a.suit === selectedSuit)), {primary:true, disabled:selectedSuit === null, className:'suit-confirm'}); return;
  }
  if (!obs.players[0].can_act) { const text=document.createElement('span'); text.className='action-description'; text.textContent='你已胡牌，剩余对手继续。'; $('actions').append(text); return; }
  const discards = obs.legal_actions.filter((a) => a.kind === 'discard');
  if (discards.length) {
    const action=discards.find((a) => a.tiles[0] === selected[0]);
    actionButton(action ? `打出${tileName(action.tiles[0])}` : '出牌', () => submitAction(action), {primary:true, disabled:!action});
  }
  if(discards.length){
    const declaration=obs.legal_actions.find(a=>a.kind==='riichi'&&a.tiles[0]===selected[0]);
    if(obs.legal_actions.some(a=>a.kind==='riichi'))actionButton(declaration ? `立直 · 打出${tileName(declaration.tiles[0])}` : '选牌后立直', ()=>submitAction(declaration), {gold:true,disabled:!declaration});
  }
  const seen=new Set();
  obs.legal_actions.filter((a) => !['discard','riichi','exchange','declare_suit'].includes(a.kind)).forEach((action) => {
    const key=`${action.kind}:${action.tiles.map((t) => Math.floor(t / 4)+(obs.red_fives?.includes(t)?'-red':'')).join(',')}:${action.meld_index}`;
    if (seen.has(key)) return; seen.add(key);
    const words={abort_nine:'九种九牌',chi:'吃',pass:'过',peng:'碰',open_kong:'杠',concealed_kong:'暗杠',added_kong:'加杠',ron:'荣和',tsumo:'自摸'};
    const suffix=action.tiles.length ? ` ${action.kind==='chi' ? [...action.tiles,obs.pending_tile].sort((a,b)=>a-b).map(tileName).join('、') : action.tiles.map(tileName).filter((x,i,a)=>a.indexOf(x)===i).join('、')}` : '';
    actionButton(words[action.kind]+suffix, () => submitAction(action), {gold:['ron','tsumo'].includes(action.kind)});
  });
}

async function submitAction(action) {
  if (busy || !action || view?.mode !== 'play') return;
  const id=view.id, decision=view.observation.decision_id, token=++generation;
  busy=true; selected=[]; render(view); tone(['ron','tsumo'].includes(action.kind) ? 'win' : 'tile');
  try {
    const result=await api(`/api/games/${id}/action`, {decision_id:decision, action});
    liveView=result.view;
    for (const frame of result.frames) {
      if (token !== generation || !animations) break;
      render(frame);
      await delay(frame.observation.phase === 'terminal' ? 180 : 240);
    }
    if (token === generation) { busy=false; render(liveView); maybeSettlement(); }
  } catch (error) {
    busy=false; toast(error.message);
    try { liveView=await api(`/api/games/${id}`); render(liveView); } catch (_) { render(view); }
  }
}

async function nextHand(){
  if(busy||!liveView?.match||liveView.match.ended)return;
  busy=true;
  try{liveView=await api(`/api/games/${liveView.id}/next`,{decision_id:liveView.observation.decision_id,hands_played:liveView.match.hands_played});selected=[];$('settlement-dialog').close();busy=false;render(liveView);maybeSettlement();}
  catch(error){busy=false;toast(error.message);render(liveView);}
}

async function startGame(settings={}) {
  if (busy) return;
  stopPlayback(); generation++; busy=true;
  $('start-game').disabled=true;
  try {
    liveView=await api('/api/games', settings); selected=[]; selectedSuit=null;
    sessionStorage.setItem('mahjong-game', liveView.id);
    $('new-dialog').close(); $('settlement-dialog').close(); $('replay-dialog').close();
    busy=false; render(liveView);
  } catch (error) { busy=false; toast(error.message); if (view) render(view); }
  finally { $('start-game').disabled=false; }
}

function showSettlement() {
  const resultView=view?.mode==='replay'?view:liveView;
  const obs=resultView?.observation;
  if (!obs?.end_reason) return;
  $('settlement-title').textContent=(obs.variant==='riichi' ? obs.scores[0] >= 25000 : obs.scores[0] > 0) ? '这一局，赢得漂亮。' : '一局落定，再战一场。';
  $('settlement-reason').textContent=(END_NAMES[obs.end_reason] || obs.end_reason)+(resultView.match?.ended ? ' 整场结束。' : '');
  $('settlement-scores').replaceChildren();
  for (let seat=0;seat<4;seat++) {
    const win=obs.wins.find((w) => w.player === seat), settlement=obs.settlement;
    const result=win ? `${win.self_draw ? '自摸' : '荣和'} · ${win.value.fan} 番` : settlement.flowers?.includes(seat) ? '花猪' : settlement.ready?.[String(seat)] ? '听牌' : settlement.noten?.includes(seat) ? '未听牌' : '未胡';
    const score=(resultView.match?.final_scores || obs.scores)[seat], node=document.createElement('div'); node.className='score-card';
    node.innerHTML=`<img src="/assets/avatar-${seat}.svg" alt=""><div class="name">${escape(resultView.names[seat])}</div><div class="total ${score<0?'negative':''}">${score>0?'+':''}${score}</div><div class="result">${result}</div>`;
    $('settlement-scores').append(node);
  }
  $('settlement-details').replaceChildren();
  obs.wins.forEach((win) => {
    const node=document.createElement('div'); node.className='settlement-line';
    node.innerHTML=`<strong>${escape(resultView.names[win.player])}${win.self_draw?'自摸':'荣和'}</strong><span>${win.value.items.map((item) => `${escape(item.name)} ${item.fan}番`).join(' + ')}${win.value.fu !== undefined ? ` · ${win.value.han}番${win.value.fu}符${win.value.limit ? ' · '+escape(win.value.limit) : ''}` : ''}${win.value.raw_fan>win.value.fan ? `，封顶 ${win.value.fan} 番` : ''}</span>`;
    $('settlement-details').append(node);
  });
  obs.events.filter((event) => event.kind === 'payment').forEach((event) => {
    const node=document.createElement('div'); node.className='settlement-line';
    node.innerHTML=`<strong>${reasonName(event.data.reason)}</strong><span>${escape(event.data.payer===4?'供托':resultView.names[event.data.payer])} → ${escape(event.data.receiver===4?'供托':resultView.names[event.data.receiver])} · ${event.data.amount} 分</span>`;
    $('settlement-details').append(node);
  });
  if (!$('settlement-details').children.length) $('settlement-details').textContent='本局没有分数转移。';
  $('download-replay').hidden=view.mode==='replay'; $('review-finished').hidden=view.mode==='replay';
  $('next-hand').hidden=view.mode==='replay'||!resultView.match || resultView.match.ended;
  $('download-match').hidden=view.mode==='replay'||!resultView.match?.ended;
  if (!$('settlement-dialog').open) $('settlement-dialog').showModal();
}

function maybeSettlement() {
  if (view?.mode === 'play' && liveView?.observation.end_reason && !announcedResults.has(liveView.id+':'+(liveView.match?.hands_played||0))) {
    announcedResults.add(liveView.id+':'+(liveView.match?.hands_played||0)); tone('win'); showSettlement();
  }
}

async function loadReplay(body) {
  if (busy) return;
  stopPlayback(); generation++; busy=true; $('replay-error').textContent='正在校验牌谱…';
  try {
    const next=await api('/api/replays', body);
    busy=false; $('replay-error').textContent=''; $('replay-dialog').close(); $('settlement-dialog').close();
    $('reveal-hands').checked=false; render(next);
  } catch (error) { busy=false; $('replay-error').textContent=`牌谱无法打开：${error.message}`; toast('牌谱校验失败，请检查文件格式与版本。'); if (view) render(view); }
}

function renderReplayControls() {
  const replay=view.replay;
  $('replay-hand').hidden=replay.hands.length<=1;
  $('replay-hand').replaceChildren();
  replay.hands.forEach((hand,index)=>{const o=document.createElement('option');o.value=hand.start;o.textContent=`${index+1}. ${hand.label}`;o.selected=replay.step>=hand.start&&(index===replay.hands.length-1||replay.step<replay.hands[index+1].start);$('replay-hand').append(o);});
  $('replay-label').textContent='牌谱回放 · 校验通过'; $('replay-slider').max=replay.total; $('replay-slider').value=replay.step;
  $('replay-step').textContent=`${replay.step} / ${replay.total}`;
  $('replay-play').textContent=playingReplay ? '暂停' : '播放';
  $('replay-play').setAttribute('aria-label', playingReplay ? '暂停牌谱' : '播放牌谱');
  $('replay-first').disabled=busy || replay.step===0; $('replay-prev').disabled=busy || replay.step===0;
  $('replay-next').disabled=busy || replay.step===replay.total; $('replay-last').disabled=busy || replay.step===replay.total;
  $('replay-slider').disabled=busy; $('return-game').disabled=!liveView || busy;
}

async function seek(step, manual=true) {
  if (!view || view.mode!=='replay') return;
  if (manual) stopPlayback();
  const id=view.id, token=++replayRequest, total=view.replay.total;
  step=Math.max(0,Math.min(total,Number(step)));
  busy=true; renderReplayControls();
  try {
    const next=await api(`/api/replays/${id}?step=${step}&reveal=${$('reveal-hands').checked}`);
    if (token===replayRequest) { busy=false; render(next); }
  } catch(error) { if(token===replayRequest){busy=false;stopPlayback();toast(error.message);renderReplayControls();} }
}

function stopPlayback() { clearTimeout(replayTimer); playingReplay=false; if(view?.mode==='replay') $('replay-play').textContent='播放'; }
async function replayTick() {
  if(!playingReplay || view?.mode!=='replay') return;
  if(view.replay.step>=view.replay.total){stopPlayback();renderReplayControls();return;}
  await seek(view.replay.step+1,false);
  if(playingReplay) replayTimer=setTimeout(replayTick,Number($('replay-speed').value));
}

function returnGame() { if(busy || !liveView) return; stopPlayback(); replayRequest++; render(liveView); }

document.querySelectorAll('[data-close]').forEach((button) => button.addEventListener('click', () => $(button.dataset.close).close()));
document.querySelectorAll('dialog').forEach((dialog) => dialog.addEventListener('click', (event) => {
  if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}
}));
$('new-game').addEventListener('click', () => $('new-dialog').showModal());
$('nav-play').addEventListener('click', returnGame);
$('nav-replay').addEventListener('click', () => $('replay-dialog').showModal());
$('nav-rules').addEventListener('click', () => $('rules-dialog').showModal());
$('rule-help').addEventListener('click', () => $('rules-dialog').showModal());
$('new-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const seed=$('seed-input').value.trim();
  if(seed && (!Number.isSafeInteger(Number(seed)) || Number(seed)<0)){toast('请输入非负整数种子。');return;}
  startGame({seed:seed?Number(seed):null,policy:$('ai-policy').value,variant:$('variant-setting').value,match:$('variant-setting').value==='riichi'?$('match-setting').value:'hand',red:$('red-setting').checked,tanyao:$('tanyao-setting').checked,exchange:$('exchange-setting').checked,auto_pass:$('auto-pass-setting').checked});
});
$('animation-toggle').addEventListener('click', () => {
  animations=!animations;localStorage.setItem('mahjong-animation',animations?'on':'off');
  $('animation-toggle').textContent=`动画${animations?'开':'关'}`;$('animation-toggle').setAttribute('aria-pressed',String(animations));
});
$('animation-toggle').textContent=`动画${animations?'开':'关'}`;$('animation-toggle').setAttribute('aria-pressed',String(animations));
$('sound-toggle').addEventListener('click', () => {
  sound=!sound;
  if(sound){
    try{audio ||= new (window.AudioContext||window.webkitAudioContext)();audio.resume();tone();}
    catch(_){sound=false;toast('当前浏览器暂不支持牌桌声音。');}
  }
  $('sound-toggle').textContent=`声音${sound?'开':'关'}`;$('sound-toggle').setAttribute('aria-pressed',String(sound));
});
$('variant-setting').addEventListener('change',()=>{const riichi=$('variant-setting').value==='riichi';$('riichi-settings').hidden=!riichi;$('exchange-field').hidden=riichi;});
$('load-riichi-example').addEventListener('click',()=>loadReplay({example:'riichi'}));
$('replay-hand').addEventListener('change',e=>seek(Number(e.target.value)));
$('next-hand').addEventListener('click',nextHand);
$('download-match').addEventListener('click',()=>{const link=document.createElement('a');link.href=`/api/games/${liveView.id}/replay?match=true`;link.download='mahjong-match.json';link.click();});
$('load-example').addEventListener('click', () => loadReplay({example:'three_winners'}));
$('replay-file').addEventListener('change', async(event) => {
  const file=event.target.files[0];if(!file)return;
  if(file.size>2*1024*1024){$('replay-error').textContent='文件超过 2 MiB，请选择单局牌谱。';return;}
  try{await loadReplay({replay:JSON.parse(await file.text())});}catch(_){$('replay-error').textContent='无法读取 JSON 文件，请检查牌谱格式。';}
  event.target.value='';
});
$('replay-current').addEventListener('click', () => loadReplay({game_id:liveView.id}));
$('review-finished').addEventListener('click', () => loadReplay({game_id:liveView.id}));
$('play-again').addEventListener('click', () => { $('settlement-dialog').close(); $('new-dialog').showModal(); });
$('download-replay').addEventListener('click', () => { const link=document.createElement('a');link.href=`/api/games/${liveView.id}/replay`;link.download='mahjong.json';document.body.append(link);link.click();link.remove(); });
$('return-game').addEventListener('click', returnGame);
$('replay-first').addEventListener('click', () => seek(0));
$('replay-prev').addEventListener('click', () => seek(view.replay.step-1));
$('replay-next').addEventListener('click', () => seek(view.replay.step+1));
$('replay-last').addEventListener('click', () => seek(view.replay.total));
$('replay-slider').addEventListener('change', (event) => seek(Number(event.target.value)));
$('reveal-hands').addEventListener('change', () => seek(view.replay.step));
$('replay-play').addEventListener('click', () => {
  if(busy)return;
  if(playingReplay){stopPlayback();renderReplayControls();return;}
  playingReplay=true;renderReplayControls();
  if(view.replay.step===view.replay.total)seek(0,false).then(() => {if(playingReplay)replayTimer=setTimeout(replayTick,0);});
  else replayTimer=setTimeout(replayTick,0);
});
for(let seat=0;seat<4;seat++)$(`river-${seat}`).addEventListener('click', () => {
  if(!view)return;
  $('river-title').textContent=`${view.names[seat]}的牌河`;$('river-detail').replaceChildren();
  view.observation.players[seat].river.forEach((discard) => $('river-detail').append(tileElement(discard.tile,{called:discard.called_by!==null,win:discard.winner_refs.length>0})));
  if(!$('river-detail').children.length)$('river-detail').textContent='尚未出牌。';
  $('river-dialog').showModal();
});
document.addEventListener('keydown', (event) => {
  if(busy||view?.mode!=='play'||document.querySelector('dialog[open]')||['INPUT','SELECT'].includes(event.target.tagName))return;
  if(event.key==='Enter' && selected.length===1 && view.observation.phase==='self_turn'){
    const action=view.observation.legal_actions.find((a) => a.kind==='discard'&&a.tiles[0]===selected[0]);
    if(action){event.preventDefault();submitAction(action);}
  }
  if(event.key==='Escape'){selected=[];renderHand();renderActions();}
});

async function initialize() {
  const existing=sessionStorage.getItem('mahjong-game');
  if(existing){try{liveView=await api(`/api/games/${existing}`);render(liveView);maybeSettlement();return;}catch(_){sessionStorage.removeItem('mahjong-game');}}
  await startGame();
}
initialize();
