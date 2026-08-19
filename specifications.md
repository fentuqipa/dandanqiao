# Professor Qiao Dandan Website — Specifications

## 1. Document purpose

This document describes the current functional, content, visual, and technical specification for Professor Qiao Dandan's academic portfolio website. It is intended to be the reference for future design, content, development, testing, and deployment work.

Status: current-state specification  
Application type: client-side single-page website  
Primary language: English  
Repository package: `professorqiaodandanwebsite`  
Configured deployment path: `/professorqiaodandanwebsite`

## 2. Product goals

The website must:

- Present Qiao Dandan's academic profile, appointments, education, and research interests.
- Provide a browsable and searchable record of journal and conference publications.
- Present teaching interests and courses.
- Provide direct access to contact, office location, school profile, and Google Scholar.
- Present searchable current-student and alumni information in a responsive card grid.
- Allow visitors to move between page sections without a full-page reload.
- Remain readable and usable on desktop, tablet, and mobile screen sizes.

The current implementation is informational only. It has no authentication, content management system, database, analytics, server API, forms, or user-generated content.

## 3. Audience

Primary audiences include:

- Academic peers and research collaborators.
- Prospective Ph.D., undergraduate, and graduate students.
- Conference and journal stakeholders.
- NUS colleagues and administrators.
- Visitors seeking publication, teaching, contact, or biographical information.

## 4. Information architecture

The application is one vertically scrolling page with six navigation targets.

| Order | Section ID | Navigation label | Current status |
| --- | --- | --- | --- |
| 01 | `about` | About | Implemented |
| 02 | `research` | Research | Implemented |
| 03 | `teaching` | Teaching | Implemented |
| 04 | `services` | Services | Implemented |
| 05 | `awards` | Awards | Implemented |
| 06 | `students` | Students | Implemented |

The Services and Awards sections are rendered and participate fully in navigation and active-section tracking.

## 5. Global layout and navigation

### 5.1 Page shell

- The main content container has a maximum width of 1,060 px, with 40 px total horizontal viewport clearance on larger screens.
- A fixed, decorative background sits behind all content.
- The background uses layered teal, purple, and peach radial gradients over a light linear gradient, plus a 32 px grid texture.
- Main content begins below the sticky navigation and ends with bottom spacing.

### 5.2 Sticky hotbar

The hotbar remains near the top of the viewport and contains:

- The current section number and label on desktop.
- A `Qiao Dandan` brand button that navigates to About.
- Buttons for About, Research, Teaching, Services, Awards, and Students.
- A search button with a search icon and a visual `⌘K` hint.

Behavior:

- The active navigation pill and left-side label reflect the section intersecting a point 35% down the viewport. At the bottom of the document, the final Students section is selected explicitly so bottom padding cannot leave the hotbar on section 05.
- Section navigation uses a custom 700 ms ease-in-out cubic scroll animation.
- Scroll targets account for a 110 px hotbar offset.
- Opening a section from global search closes the search overlay.
- The `⌘K` text is currently a visual hint only; no keyboard listener opens search.

### 5.3 Responsive navigation

- At 1,080 px and below, the left-side current-section indicator is hidden so the six section buttons have the full hotbar width.
- Hotbar buttons never shrink, the `Qiao Dandan` brand remains on one line, and the navigation row becomes horizontally scrollable when it does not fit.
- At 620 px and below, navigation buttons use reduced horizontal padding and the `⌘K` text is hidden while the search icon remains.

## 6. About section

The About section is the introductory hero and must contain:

- A circular profile image loaded from `public/profile.jpg` with alt text `Qiao Dandan`.
- A status pill identifying the subject as Associate Professor, Information Systems & Analytics at the National University of Singapore.
- The main page heading `Qiao Dandan`.
- Position history at the National University of Singapore.
- Education at Tsinghua University, The University of Texas at Austin, and Beijing University of Posts and Telecommunications.
- Research topics and methodologies.

The action row contains:

| Action | Behavior |
| --- | --- |
| Research | Smooth-scrolls to the Research section |
| COM2-04-15 | Opens the office location in Google Maps in a new tab |
| School Page | Opens the NUS School of Computing profile in a new tab |
| qiaodd@nus.edu.sg | Opens the visitor's email client using `mailto:` |
| Google Scholar | Opens the Google Scholar profile in a new tab |

The section is centered at the hero level, while biography lists are left-aligned. Action controls remain in a single flex row and may rely on available viewport width.

## 7. Research and publications

### 7.1 Publication categories

The Research section contains four source categories:

- Journal publications, with an optional external publication URL.
- Conference publications, currently without external URLs.
- Papers under review.
- Working papers.

Each publication record contains:

- `year`: a string used for display and chronological sorting.
- `text`: formatted React content containing authors, title, venue, and citation details.
- `href`: optional external URL.

All four publication arrays use the shared `TimelinePublication` type and are maintained in the central content module, `src/data/siteContent.tsx`. Formatted React fragments are allowed in a publication's `text` field so author emphasis and italicized venues remain editable with the citation.

### 7.2 Category tabs

The publication controls provide four tabs:

- **All**: merges every populated publication category into one timeline and sorts them by numeric year in descending order. For matching years, existing source order is preserved. Every card in this view displays its category in a small top-left tag.
- **Journals**: displays journal publications only.
- **Conferences**: displays conference publications only.
- **Working Papers**: displays current working papers.

Papers under review do not have a dedicated tab and remain available through the All tab. The category tags appear only in the All tab; cards in individual category tabs do not display them. The active tab uses a dark background and exposes its state through `aria-selected`.

### 7.3 Publication search

The inline publication search:

- Is controlled by React component state.
- Filters results immediately as the visitor types.
- Is case-insensitive and trims leading and trailing whitespace.
- Matches contiguous text within publication title, author, and venue content.
- Includes italicized journal and conference venue names, allowing searches such as a journal title, conference title, or venue acronym when the acronym appears in the citation.
- Applies within the currently active category tab.
- Displays `No publications are available in this category yet.` when an empty category is opened without a search query.
- Displays `No publications match “{query}”.` when there are no results.

This search is independent from the global site search overlay.

### 7.4 Result limiting and expansion

- A maximum of 10 matching publications is displayed initially.
- If the active filtered collection contains more than 10 entries, a centered `Show More Publications` button appears after the timeline.
- Activating `Show More Publications` reveals every matching publication and changes the control to `Hide Extra Publications` in the same centered location.
- Activating `Hide Extra Publications` restores the 10-item view, shortens the section, and changes the control back to `Show More Publications`.
- Expansion state persists while the Research component remains mounted, including after tab or query changes.

### 7.5 Publication timeline and links

- Results appear as cards in a vertical timeline.
- Each row displays the year in both the timeline marker and the card.
- In the All tab, each card's top-left category tag identifies its source category; the currently populated Journal and Conference categories use separate accent colors.
- Publication cards with an `href` are full-card links and open in a new tab with `rel="noreferrer"`.
- Publications without an `href` render as non-interactive articles.
- On screens at or below 700 px, the vertical line and dot markers are hidden and rows become single-column cards.

## 8. Teaching, Services, Awards, and Students

The Teaching section contains two content groups.

### 8.1 Teaching interests

- Business Analytics.
- Platform Design.
- Computational Social Science.

### 8.2 Class Instructor / Teaching Assistant

- Computational Social Science (`IS6006`), Ph.D. level, National University of Singapore.
- Mining Business Insights from Web Data (`BT4222`), undergraduate level, National University of Singapore.
- Analytics Driven Design of Adaptive Systems (`BT4014`), undergraduate level, National University of Singapore.

Teaching content is maintained under `siteContent.teaching.groups` in `src/data/siteContent.tsx`.

### 8.3 Services and Awards

- Services renders as section `04` with the `services` ID and Services-specific layout classes.
- Awards renders as section `05` with the `awards` ID and Awards-specific layout classes.
- Services presents university service, journal and conference editorship, and journal and conference referee activity.
- Awards presents grants and academic or professional awards.
- Both are mounted by `App.tsx`, receive section refs, support hotbar navigation, and participate in active-section detection.
- Teaching, Services, and Awards size to their content rather than enforcing a full viewport height. Adjacent sections use 92 px of combined boundary padding to maintain separation without large empty gaps.
- Teaching, Services, and Awards use a compact responsive type scale: group headings range from `1rem` to `1.25rem`, while list content ranges from `0.9rem` to `1.05rem`.

### 8.4 Students

- Students renders as section `06` with the `students` ID and participates in hotbar navigation and active-section detection.
- The section contains two permanently visible subsections in order: **Alumni** and **Students**.
- Alumni records are maintained in the top-level `alumniItems` array, while current student records are maintained in the separate top-level `studentItems` array in `src/data/siteContent.tsx`.
- Each record has a `name`, `description`, numeric `startYear`, optional numeric `endYear`, optional `picture`, optional `pictureAlt`, and optional `profileUrl`; descriptions may contain line breaks. Array membership determines whether the record is an alumnus or current student.
- Each subsection has its own page-centered heading, search input, responsive card grid, and empty state. There are no status tabs or status tags.
- Both grids render in three columns on desktop.
- Cards display `startYear–endYear` when an end year is present and display only `startYear` otherwise.
- Each card displays a centered circular picture above the name. Records without a `picture` use `public/default-profile.svg`, a neutral grey profile placeholder.
- When `profileUrl` is present, the entire card is a link that opens the profile in a new tab with `rel="noreferrer"`. Linked cards retain the same typography as non-linked cards and use a slightly extended horizontal hover/focus highlight. Cards without a profile URL remain non-interactive articles.
- Alumni are sorted by `endYear` in descending order; records without an end year appear last, and Alumni sharing an end year are sorted alphabetically by name. Current students are sorted by `startYear` in ascending order; Students sharing a start year retain their source order from `studentItems`.
- Each search independently filters its subsection immediately and case-insensitively across names and descriptions. Multi-word queries use order-independent AND matching, so a name can be found in either given-name/surname or surname/given-name order.
- With no matching records, the corresponding grid displays a query-specific or subsection-specific empty-state message.
- The section has a 70% viewport minimum height so the final navigation target can become active even when a filtered result set is empty.

## 9. Global site search

### 9.1 Opening and closing

- The hotbar search control opens a modal-style search overlay.
- Clicking the backdrop closes the overlay.
- Clicking the visible `ESC` button closes the overlay.
- Clicking inside the search panel must not close it.
- The search input receives focus when the overlay opens.
- Pressing the Escape key is not currently implemented.
- The query remains in component state after closing and reopening the overlay.

### 9.2 Matching

The global search index is generated automatically by `SearchOverlay.tsx` from:

- The About, Teaching, Services, Awards, and Students content objects in `siteContent`.
- Every record in the top-level `alumniItems` and `studentItems` arrays.
- Every record in `journalPublications`, `conferencePublications`, `underReviewPublications`, and `workingPaperPublications`.

There is no manually maintained `searchItems` array. React-formatted content is converted to plain searchable text while retaining its displayed formatting at the original section.

Generated results include section overviews, individual content groups, individual students, and individual publications. Each entry receives a stable destination section, generated display metadata, and the corresponding section color.

Matching behavior:

- Matching is case-insensitive.
- Leading and trailing whitespace is ignored.
- Multiple query words use AND semantics: every word must occur somewhere in the generated title, subtitle, group, or source-content text.
- An empty query returns the six section-overview entries rather than the full publication and student collections.
- Results are grouped by section or publication category.
- Selecting a result navigates to its parent section, not to an individual subsection or publication.
- When no entries match, the overlay displays suggested example searches.

### 9.3 Overlay presentation

- The backdrop covers the viewport and applies blur over a translucent dark layer.
- The panel is limited to 820 px wide and to the available viewport height.
- Results scroll inside the panel when necessary.
- Each result displays a colored dot, title, optional badge, subtitle, and a visual return-key hint.

## 10. Visual design system

### 10.1 Typography

- Primary application typography uses `Inter` when locally available, followed by system UI fallbacks.
- No web font is downloaded.
- Major titles use responsive `clamp()` sizing and tight negative letter spacing.
- The About hero name uses a responsive `3.4rem–6.2rem` desktop scale and a reduced `2.4rem–3.4rem` mobile scale.
- Search metadata uses system monospace fonts.

### 10.2 Color and surfaces

- Primary text is near-black with gray secondary copy.
- Blue/purple identifies Research and active timeline details.
- Teal is used for section status and supporting accents.
- Cards use translucent white backgrounds, subtle borders, large rounded corners, blur, and soft shadows.
- Active publication tabs and the publication expansion button use dark navy (`#151a2a`) with white text.

### 10.3 Motion and interaction

- Navigation scrolling is animated.
- Interactive cards and buttons use short hover translations and shadow/background transitions.
- Focus-visible styles are defined for publication tabs and the expansion button.
- The current stylesheet does not include a `prefers-reduced-motion` override.

### 10.4 Responsive breakpoints

| Breakpoint | Main changes |
| --- | --- |
| `max-width: 1080px` | Hotbar indicator hides; navigation receives the full row and scrolls horizontally without shrinking buttons |
| `max-width: 1000px` | Publication search and tabs stack; tabs may wrap to additional lines |
| `max-width: 900px` | Multi-column content grids collapse |
| `max-width: 700px` | Publication tabs use horizontal scrolling; timeline simplifies; publication cards become single-column |
| `max-width: 620px` | Main margins and navigation padding reduce; selected multi-column cards collapse |
| `max-width: 380px` | About action buttons switch from two columns to one full-width control per row |

Both Students-section grids display three columns above 900 px, two columns at 900 px and below, and one column at 620 px and below. Each search bar remains full-width above its corresponding grid.

At 620 px and below, mobile-only rules preserve the desktop presentation while adapting the site as follows:

- The hotbar remains horizontally scrollable with compact, non-shrinking, touch-sized buttons.
- The About hero uses reduced image, title, and vertical spacing; its action buttons wrap into two columns and then one column below 380 px.
- About, Teaching, Services, Awards, and Students content uses the full available main-column width.
- Publication cards use compact single-column spacing, and the publication expansion control fills the available width.
- Long service, award, and teaching list content may wrap anywhere rather than causing horizontal overflow.
- The global search panel fits the dynamic viewport, uses compact padding, and hides decorative keyboard hints to leave room for result text.

The minimum supported layout width is 320 px as declared by the global body styles.

## 11. Accessibility requirements and current behavior

Implemented accessibility provisions include:

- Semantic `header`, `nav`, `main`, `section`, `article`, heading, list, button, link, and input elements.
- Descriptive profile-image alt text.
- `aria-label` on page navigation and search controls.
- Tab roles and `aria-selected` state for publication filtering.
- A screen-reader-only label for publication search.
- A screen-reader-only label for each Alumni/Students search and a polite live region for each result grid.
- `role="status"` on the publication no-results message.
- Keyboard-focus styles on publication controls.
- Native buttons and links for interactive elements.

Known accessibility gaps:

- The global search overlay does not declare dialog semantics, `aria-modal`, or a programmatic label.
- Focus is not trapped in the open search overlay and is not restored explicitly on close.
- Escape-key handling and the displayed `⌘K` shortcut are not implemented.
- Custom animated scrolling does not honor reduced-motion preferences.
- Publication tab elements do not implement arrow-key tablist navigation or `aria-controls` relationships.
- The purely visual About status dot is not explicitly hidden from assistive technology.

## 12. Technical architecture

### 12.1 Runtime stack

- React 19.
- React DOM 19.
- TypeScript 6.
- Vite 8 with `@vitejs/plugin-react`.
- `lucide-react` for interface icons.
- No router; navigation uses section IDs and scrolling.
- No runtime state management dependency; local React state and refs are sufficient.

### 12.2 Component responsibilities

| File | Responsibility |
| --- | --- |
| `src/main.tsx` | Mounts the React application in Strict Mode |
| `src/App.tsx` | Composes the page, owns navigation/search state, tracks active sections, and performs animated scrolling |
| `src/components/Hotbar.tsx` | Sticky navigation and global-search trigger |
| `src/components/SearchOverlay.tsx` | Generates the global index from central content and owns query matching, grouping, and result rendering |
| `src/components/AboutSection.tsx` | Renders the academic profile and external/contact actions from central content |
| `src/components/ResearchSection.tsx` | Research section presentation, category configuration, filtering, inline search, limiting, expansion, and timeline rendering |
| `src/components/TeachingSection.tsx` | Renders teaching interests and course history from central content |
| `src/components/ServicesSection.tsx` | Renders university service, editorship, and referee activity from central content |
| `src/components/AwardsSection.tsx` | Renders grants and academic or professional awards from central content |
| `src/components/StudentsSection.tsx` | Independent Alumni and Students searches, empty states, and responsive grids |
| `src/components/SectionShell.tsx` | Generic section wrapper; currently not used by `App.tsx` |
| `src/data/siteContent.tsx` | Central data source for the document title, navigation, non-Research section copy, external links, publication arrays, students, and shared UI labels |
| `src/styles/global.css` | Active application layout, component, interaction, and responsive styling |

### 12.3 State ownership

`App.tsx` owns:

- Current active section.
- Global search overlay visibility.
- Element refs for every navigation section.

`SearchOverlay.tsx` owns:

- Global search query.

`ResearchSection.tsx` owns:

- Active publication category.
- Publication query.
- Whether the full publication collection has been expanded.

`StudentsSection.tsx` owns:

- Independent Alumni and Students search queries. Editable records remain in the central content module.

All state is transient and resets on a full page refresh.

### 12.4 Central content storage

`src/data/siteContent.tsx` is the central source for academic content and records. The Research section keeps its presentation, interaction logic, filter labels, headings, and other interface copy in `src/components/ResearchSection.tsx`; only its publication datasets live in the central content module.

The central module contains:

- `siteContent.siteTitle` and `siteContent.brandLabel` for the browser title and hotbar brand.
- `siteContent.navigation` for ordered section labels and IDs.
- `siteContent.about` for the name, portrait metadata, position, education, research interests, action labels, contact details, and external URLs.
- `siteContent.teaching.groups`, `siteContent.services.groups`, and `siteContent.awards.groups` for their headings and list content.
- `siteContent.students` for section and subsection labels, search copy, and empty-state messages.
- `alumniItems` and `studentItems` for the separate Alumni and current-student record collections.
- `journalPublications`, `conferencePublications`, `underReviewPublications`, and `workingPaperPublications` for publication records.
- `siteContent.search` for global-search interface copy; searchable entries are derived automatically from the other content.

The file uses a `.tsx` extension because some About details and publication citations contain React fragments for bold and italic formatting. Content remains statically bundled; there is no CMS, database, runtime content fetch, or schema-validation service.

### 12.5 Content editing rules

- Edit general site text, records, links, section labels, and the browser title in `src/data/siteContent.tsx`.
- Edit Research section interface copy, filter labels, headings, result limits, and interaction behavior in `src/components/ResearchSection.tsx`.
- Keep every navigation `id` within the `SectionId` union and synchronized with its rendered section ID.
- Student and alumni descriptions may contain `\n` line breaks. Every record requires `startYear`; `endYear`, `picture`, `pictureAlt`, and `profileUrl` may be omitted. Move a record between `alumniItems` and `studentItems` to change its subsection.
- Publication `year` remains a string, `text` contains the formatted citation, and `href` is optional.
- Add a publication to exactly one source array. The All view is generated automatically.
- New section groups, students, and publications become searchable automatically; no separate search-index maintenance is required.
- Components in `src/components` are presentation and interaction code. `ResearchSection.tsx` also owns Research-specific interface labels, but publication records must remain in `siteContent.tsx`.

## 13. Assets and external destinations

Local public assets:

- `public/profile.jpg`: About portrait.
- `public/favicon.svg`: browser favicon.
- `public/icons.svg`: additional public icon asset not currently referenced by the rendered components.

External destinations:

- Google Maps office location.
- NUS School of Computing profile.
- Google Scholar profile.
- `mailto:qiaodd@nus.edu.sg`.
- Journal publication pages across publisher and research-hosting domains.

External HTTP links must open in a new tab where currently specified and must retain safe `rel` attributes.

## 14. Build, quality, and deployment

### 14.1 Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite development server |
| `npm run build` | Type-check with project references and create a production Vite build |
| `npm run lint` | Run ESLint across the project |
| `npm run preview` | Preview the production build locally |
| `npm run deploy` | Publish `dist` using `gh-pages` |

On Windows systems where PowerShell script execution blocks `npm.ps1`, the equivalent `npm.cmd run <script>` command may be used.

### 14.2 Build output and hosting

- Production output is generated in `dist/`.
- Vite's base path is `/professorqiaodandanwebsite`.
- The package homepage targets `https://leezehao.github.io/professorqiaodandanwebsite/`.
- Deployment uses the `gh-pages` package.
- The application must remain compatible with static hosting and must not require server-side URL rewriting.

### 14.3 TypeScript and browser target

- Compilation targets ES2023 and DOM APIs.
- JSX uses the automatic React JSX transform.
- Module resolution uses bundler mode.
- Unused locals and parameters are TypeScript errors.
- Emission is handled by Vite; TypeScript performs type-checking only.

## 15. Acceptance criteria

A release satisfies the current specification when:

1. All six hotbar targets scroll to their corresponding section and the active item updates while scrolling.
2. About displays the portrait, academic biography, research interests, and all five actions.
3. The All publication tab renders one merged, descending-by-year timeline.
4. Journal, Conference, and Working Papers tabs show only their respective publication types; papers under review remain visible through the All tab.
5. Publication search filters the active category by title, author, or venue without a page reload.
6. At most 10 publications appear before expansion, `Show More Publications` reveals all matching entries, and `Hide Extra Publications` restores the limited view.
7. Linked publication cards open the correct external URL in a new tab.
8. Global search filters configured search entries using case-insensitive AND matching and navigates to the selected section.
9. Alumni and current students render from separate central arrays in two independent three-column desktop grids; Alumni sort by descending end year with alphabetical ties, Students sort by ascending start year with same-year source order preserved, and each subsection search matches names, descriptions, and years.
10. The layout remains usable at desktop, tablet, and 320 px mobile widths.
11. `npm run build` and `npm run lint` complete successfully.
12. The production build works from the configured GitHub Pages base path.

## 16. Known gaps and maintenance notes

- The document title is set from `siteContent.siteTitle`; no description, social-sharing metadata, canonical URL, or structured data is configured.
- Global search navigates generated student and publication results to their parent section rather than scrolling to or highlighting the exact record.
- Publication search derives text from rendered citation nodes; a structured publication model with explicit `authors`, `title`, `venue`, `type`, and `year` fields would make future filtering and maintenance more reliable.
- `src/index.css` and `src/App.css` retain Vite starter styles. `src/index.css` is imported before the active global stylesheet, while `src/App.css` is not imported. Consolidation would reduce conflicting or obsolete styles.
- `SectionShell.tsx` and several generic global style groups are currently unused.
- There is no automated component, interaction, visual-regression, or end-to-end test suite.
- Static content contains several spelling and capitalization inconsistencies that should be reviewed separately from functional changes.
