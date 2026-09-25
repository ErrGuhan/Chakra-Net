-- =========================================================================
-- ChakraNet Server API Supabase PostgreSQL Database Schema
-- SIH 2026 PS 26078: Extreme Weather Anomaly Tracker & 5 km Downscaling Pipeline
-- =========================================================================

-- 1. Tracked Anomaly Events Table
create table if not exists public.events (
    id text primary key,
    name text not null,
    basin text not null,
    landfall jsonb not null,
    lead_times_hours jsonb not null,
    resolution jsonb not null,
    status text not null,
    updated_at timestamptz default timezone('utc'::text, now()) not null
);

comment on table public.events is 'Tracked extreme weather events (e.g. Cyclone Phailin)';

-- 2. 4D Storm Trajectories & Uncertainty Cones Table
create table if not exists public.tracks (
    event_id text primary key references public.events(id) on delete cascade,
    geojson jsonb not null,
    total_features integer not null default 0,
    updated_at timestamptz default timezone('utc'::text, now()) not null
);

comment on table public.tracks is '4D storm track line, centroid points, uncertainty cone, and bounding boxes';

-- 3. Downscaled 5 km Hazard Grids Table
create table if not exists public.hazard_grids (
    id text primary key, -- Composite key: event_id_leadtime_threshold (e.g. phailin_2013_108_75)
    event_id text not null references public.events(id) on delete cascade,
    lead_time_hours integer not null,
    threshold_mm double precision not null,
    geojson jsonb not null,
    total_cells integer not null default 0,
    updated_at timestamptz default timezone('utc'::text, now()) not null
);

create index if not exists idx_hazard_grids_lookup 
on public.hazard_grids (event_id, lead_time_hours, threshold_mm);

comment on table public.hazard_grids is '5 km downscaled probability-of-exceedance grid cells with 4D column heights';

-- 4. CAP 1.2 Alerts & Multilingual Translations Table
create table if not exists public.alerts (
    id text primary key, -- Composite key: event_id_leadtime (e.g. phailin_2013_108)
    event_id text not null references public.events(id) on delete cascade,
    lead_time_hours integer not null,
    severity text not null,
    affected_area text not null,
    cap_xml text not null,
    multilingual jsonb not null,
    updated_at timestamptz default timezone('utc'::text, now()) not null
);

comment on table public.alerts is 'OASIS CAP 1.2 XML alerts and bilingual Bhashini translations';

-- 5. Dispatches & Audit Log Table
create table if not exists public.dispatches (
    dispatch_id text primary key,
    event_id text not null,
    cell_id text,
    channels jsonb not null,
    target_districts jsonb not null,
    simulation_mode boolean not null default true,
    status text not null,
    sachet_status text not null,
    bhashini_status text not null,
    disclaimer text not null,
    audit_log jsonb not null,
    created_at timestamptz default timezone('utc'::text, now()) not null
);

create index if not exists idx_dispatches_event_id 
on public.dispatches (event_id);

comment on table public.dispatches is 'Simulated NDMA SACHET / Bhashini alert dispatch audit receipts';

-- Enable Row Level Security (RLS) on all tables
alter table public.events enable row level security;
alter table public.tracks enable row level security;
alter table public.hazard_grids enable row level security;
alter table public.alerts enable row level security;
alter table public.dispatches enable row level security;

-- Setup Permissive Read Policies (Public / Anon Access)
drop policy if exists "Allow public read access on events" on public.events;
create policy "Allow public read access on events" on public.events for select using (true);

drop policy if exists "Allow public read access on tracks" on public.tracks;
create policy "Allow public read access on tracks" on public.tracks for select using (true);

drop policy if exists "Allow public read access on hazard_grids" on public.hazard_grids;
create policy "Allow public read access on hazard_grids" on public.hazard_grids for select using (true);

drop policy if exists "Allow public read access on alerts" on public.alerts;
create policy "Allow public read access on alerts" on public.alerts for select using (true);

drop policy if exists "Allow public read access on dispatches" on public.dispatches;
create policy "Allow public read access on dispatches" on public.dispatches for select using (true);

-- Setup Service Role Full Access Policies (Service Secret Key can insert, update, delete)
drop policy if exists "Allow service role full access on events" on public.events;
create policy "Allow service role full access on events" on public.events for all using (true) with check (true);

drop policy if exists "Allow service role full access on tracks" on public.tracks;
create policy "Allow service role full access on tracks" on public.tracks for all using (true) with check (true);

drop policy if exists "Allow service role full access on hazard_grids" on public.hazard_grids;
create policy "Allow service role full access on hazard_grids" on public.hazard_grids for all using (true) with check (true);

drop policy if exists "Allow service role full access on alerts" on public.alerts;
create policy "Allow service role full access on alerts" on public.alerts for all using (true) with check (true);

drop policy if exists "Allow service role full access on dispatches" on public.dispatches;
create policy "Allow service role full access on dispatches" on public.dispatches for all using (true) with check (true);
