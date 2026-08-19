import type { ReactNode } from "react";

export type SectionId =
  | "about"
  | "research"
  | "teaching"
  | "services"
  | "awards"
  | "students";

export type ContentGroup = {
  title: string;
  details: ReactNode[];
};

export const siteContent = {
  siteTitle: "Dandan Qiao",
  brandLabel: "Qiao Dandan",
  navigation: [
    { id: "about", label: "About" },
    { id: "research", label: "Research" },
    { id: "teaching", label: "Teaching" },
    { id: "services", label: "Services" },
    { id: "awards", label: "Awards" },
    { id: "students", label: "Students" },
  ] satisfies Array<{ id: SectionId; label: string }>,
  about: {
    portrait: { src: "profile.jpg", alt: "Qiao Dandan" },
    status:
      "Associate Professor, Information Systems & Analytics @ National University of Singapore",
    name: "Dandan Qiao",
    groups: [
      {
        title: "Position",
        details: [
          "Associate Professor, Information Systems & Analytics, National University of Singapore, 2024.07–Present",
          "Assistant Professor, Information Systems & Analytics, National University of Singapore, 2018.07–2024.06",
        ],
      },
      {
        title: "Education",
        details: [
          "Ph.D., Information Systems, Tsinghua University, 2018",
          "Visiting Student, Information Systems, The University of Texas at Austin, 2015.08–2017.09",
          "B.S., Information Systems, Beijing University of Posts and Telecommunications, 2012",
        ],
      },
      {
        title: "Research Interests",
        details: [
          <>
            <b>Topics:</b> Digital Platform Design, AI and Business Application,
            Computational Social Science
          </>,
          <>
            <b>Methodologies:</b> Computational Methods, Econometric Analysis,
            Field Experiments
          </>,
        ],
      },
    ] satisfies ContentGroup[],
    actions: {
      research: "Research",
      location: {
        label: "COM2-04-15",
        href: "https://www.google.com/maps/place/COM+2/@1.2942824,103.7741012,15z/data=!4m6!3m5!1s0x31da1af8b4dca745:0x138ac77cdd89a8f1!8m2!3d1.2942824!4d103.7741012!16s%2Fg%2F11c6lw2s04?entry=ttu",
      },
      schoolPage: {
        label: "School Page",
        href: "https://www.comp.nus.edu.sg/disa/people/qiaodd/",
      },
      email: {
        label: "qiaodd@nus.edu.sg",
        href: "mailto:qiaodd@nus.edu.sg",
      },
      scholar: {
        label: "Google Scholar",
        href: "https://scholar.google.com/citations?user=2ESyqbEAAAAJ&hl=en",
      },
    },
  },
  teaching: {
    sectionNumber: "03",
    sectionTitle: "Teaching",
    groups: [
      {
        title: "Teaching Interests",
        details: [
          "Business Analytics",
          "Platform Design",
          "Computational Social Science",
        ],
      },
      {
        title: "Class Instructor / Teaching Assistant",
        details: [
          "Computational Social Science (IS6006), PhD level, National University of Singapore",
          "Mining Business Insights from Web Data (BT4222), Undergraduate level, National University of Singapore",
          "Analytics Driven Design of Adaptive Systems (BT4014), Undergraduate level, National University of Singapore",
        ],
      },
    ] satisfies ContentGroup[],
  },
  services: {
    sectionNumber: "04",
    sectionTitle: "Services",
    groups: [
      {
        title: "Department / School / University Service",
        details: [
          "Deputy Director, Centre for Computational Social Science and Humanities (https://cssh.nus.edu.sg/), NUS, 2024 - 2026 June",
          "Ph.D. coordinator, Department of Information Systems and Analytics, NUS, 2023 Jan - 2026 June",
          "MComp IS committee member, Department of Information Systems and Analytics, NUS, 2020-2022",
        ],
      },
      {
        title: "Editorship of Journals / Conferences",
        details: [
          "Associate Editor at Management Science, 2026-Present",
          "Associate Editor at MIS Quarterly, 2026-Present",
          "Associate Editor at Decision Sciences Journal, 2026-Present",
          "Editorial Review Board Member of Information Systems Research, 2024-2025",
          "Associate Editor for Special Issue (AI-IA Nexus) of MIS Quarterly",
          "Cluster Chair for eBusiness Community of INFORMS Annual Meeting, 2024",
          "Co-organizer of Feeder Workshop for Workshop on Information Technologies and Systems, 2024",
          "Track Co-Chair for Pacific Asia Conference on Information Systems, 2023",
          "Track Associate Editor for International Conference on Information Systems, 2024, 2023, 2022, 2021, 2020, 2019",
          "Track Associate Editor for Pacific Asia Conference on Information Systems, 2024, 2022, 2021, 2020",
        ],
      },
      {
        title: "Referee for Journals",
        details: [
          "Management Science (MS)",
          "Information Systems Research (ISR)",
          "MIS Quarterly (MISQ)",
          "INFORMS Journal on Computing (JOC)",
          "Journal of Management Information Systems (JMIS)",
          "Production and Operations Management (POM)",
          "Journal of the Association for Information Science and Technology (JAIST)",
        ],
      },
      {
        title: "Referee for Conferences",
        details: [
          "International Conference on Information Systems (ICIS)",
          "Conference on Information Systems and Technology (CIST)",
          "Hawaii International Conference on System Sciences (HICSS)",
          "Workshop on Information Technologies and Systems (WITS)",
          "Pacific Asia Conference on Information Systems (PACIS)",
          "China Summer Workshop on Information Management (CSWIM)",
        ],
      },
    ] satisfies ContentGroup[],
  },
  awards: {
    sectionNumber: "05",
    sectionTitle: "Awards",
    groups: [
      {
        title: "Grants",
        details: [
          "Ministry of Education Academic Research Fund Tier-2, 2025.06 ~ 2028.06",
          "Ministry of Education Academic Research Fund Tier-1, 2024.01 ~ 2026.12",
          "Ministry of Education Academic Research Fund Start-up",
          "NET Institute Summer Grant, Summer 2019",
        ],
      },
      {
        title: "Awards",
        details: [
          "Robert Brown Promising Researcher Award, 2025",
          "AIS Early Career Award, 2024",
          "INFORMS ISS Gordon Davis Young Scholar Award, 2023",
          "17th Beijing Philosophy and Social Sciences Outstanding Achievement Awards, 2019",
          "Outstanding Graduate of Beijing, 2018, 2012",
          "Outstanding Doctoral Thesis, 2018",
          "China National Scholarship, 2016, 2009",
          "Tsinghua-Jiangzhen Scholarship, 2015, 2014",
        ],
      },
    ] satisfies ContentGroup[],
  },
  students: {
    sectionNumber: "06",
    sectionTitle: "Students",
    subsections: {
      alumni: {
        heading: "Alumni",
        searchLabel: "Search alumni by name or description",
        searchPlaceholder: "Search alumni...",
        noMatchesPrefix: "No alumni match",
        emptyMessage: "No alumni are available yet.",
      },
      students: {
        heading: "Students",
        searchLabel: "Search students by name or description",
        searchPlaceholder: "Search students...",
        noMatchesPrefix: "No students match",
        emptyMessage: "No students are available yet.",
      },
    },
  },
  search: {
    placeholder: "Search...",
    closeLabel: "ESC",
    resultKey: "↵",
    emptyMessage:
      "No matches found. Try “publications”, “AI”, “teaching”, or “contact”.",
  },
  hotbar: {
    navigationLabel: "Page sections",
    openSearchLabel: "Open search",
    searchShortcut: "⌘K",
  },
} as const;

export const navItems = siteContent.navigation;

export type PublicationCategory =
  | "Journal"
  | "Conference"
  | "Under Review"
  | "Working Paper";

export type TimelinePublication = {
  year: string;
  text: ReactNode;
  href?: string;
  category?: PublicationCategory;
};

export const journalPublications: TimelinePublication[] = [
  {
    year: "2026",
    text: (
      <>
        Jinghui Zhang, Mochen Yang, <b>Dandan Qiao</b>, Qiang Wei. "Regurgitative Training: The Value of Real Data in Training Large Language Models". <i>Management Science</i>, Forthcoming.
      </>
    ),
    href: "https://pubsonline.informs.org/doi/10.1287/mnsc.2024.07005",
  },
  {
    year: "2026",
    text: (
      <>
        <b>Dandan Qiao</b>, Huaxia Rui, Qian Xiong. (2026). "AI and Jobs: Has the Inflection Point Arrived? Evidence from an Online Labor Platform". <i>Journal of Management Information Systems</i>, 43(1), 5-37.
      </>
    ),
    href: "https://www.tandfonline.com/doi/abs/10.1080/07421222.2025.2602400",
  },
  {
    year: "2025",
    text: (
      <>
        Ying Lu, <b>Dandan Qiao</b>, Shu He, Bernard C Y Tan. (2025). “How Search Technology Breeds Illegal Transactions: Empirical Evidence from the Darknet”, <i>Management Science</i>. 72(5), 4319-4340..
      </>
    ),
    href: "https://pubsonline.informs.org/doi/10.1287/mnsc.2022.04133",
  },
  {
    year: "2024",
    text: (
      <>
        Lin Qiu, <b>Dandan Qiao</b>, Bernard CY Tan, and Andrew B. Whinston. (2024). "Leading the Horse to Water? Investigating the Impact of Ride-hailing Services on Hate Crimes", <i>Production and Operations Management</i>, 33(1), 342-363.
      </>
    ),
    href: "https://journals.sagepub.com/doi/abs/10.1177/10591478231224944",
  },
  {
    year: "2024",
    text: (
      <>
        Warut Khern-am-nuai, Hossein Ghasemkhani, <b>Dandan Qiao</b>, Karthik Natarajan Kannan. (2024) “The Impact of Online Q&As on Product Sales: The Case of Amazon Answer”, <i>Information Systems Research</i>, 35(2), 747-765.
      </>
    ),
    href: "https://pubsonline.informs.org/doi/10.1287/isre.2023.1233",
  },
  {
    year: "2024",
    text: (
      <>
        Jason Chan, Shu He, <b>Dandan Qiao</b>. Andrew B. Whinston. (2024). “Shedding Light on the Dark: The Impact of Legal Enforcement on Darknet Transactions”, <i>Information Systems Research</i>, 35(1), 145-164.
      </>
    ),
    href: "https://pubsonline.informs.org/doi/10.1287/isre.2023.1222",
  },
  {
    year: "2023",
    text: (
      <>
        <b>Dandan Qiao</b>, & Huaxia Rui. (2023) “Text Performance on the Vine Stage? The Effect of Incentive on Product Review Text Quality”. <i>Information Systems Research</i>, 34(2), pp.676-697.
      </>
    ),
    href: "https://pubsonline.informs.org/doi/abs/10.1287/isre.2022.1146",
  },
  {
    year: "2021",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, (2021). “Mitigating the Adverse Effect of Monetary Incentives on Voluntary Contributions Online”. <i>Journal of Management Information Systems</i>, 38(1), pp.82-107.
      </>
    ),
    href: "https://www.tandfonline.com/doi/full/10.1080/07421222.2021.1870385",
  },
  {
    year: "2020",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, (2020). “Financial Incentives Dampen Altruism in Online Prosocial Contributions: A Study of Online Reviews”. <i>Information Systems Research</i>, 31(4), pp.1361-1375.
      </>
    ),
    href: "https://pubsonline.informs.org/doi/10.1287/isre.2020.0949",
  },
  {
    year: "2020",
    text: (
      <>
        Liye Wang, Jin Zhang, Guoqing Chen, <b>Dandan Qiao</b>. (2020). “Identifying Comparable Entities with Indirectly Associative Relations and Word Embeddings from Web Search Logs”. <i>Decision Support Systems</i>, p.113465.
      </>
    ),
    href: "https://doi.org/10.1016/j.dss.2020.113465",
  },
  {
    year: "2017",
    text: (
      <>
        Xunhua Guo, Qiang Wei, Guoqing Chen, Jin Zhang, <b>Dandan Qiao</b>. (2017). “Extracting Representative Information on Intra-Organizational Blogging Platforms”. <i>MIS Quarterly</i>, 41(4), pp.1105-1127.
      </>
    ),
    href: "https://misq.umn.edu/extracting-representative-information-on-intra-organizational-blogging-platforms.html",
  },
  {
    year: "2017",
    text: (
      <>
        <b>Dandan Qiao</b>, Jin Zhang, Qiang Wei, Guoqing Chen. (2017). “Finding Competitive Keywords from Query Logs to Enhance Search Engine Advertising”. <i>Information & Management</i>, 54(4), pp.531-543.
      </>
    ),
    href: "https://www.sciencedirect.com/science/article/abs/pii/S0378720616303330",
  },
  {
    year: "2016",
    text: (
      <>
        Qiang Wei, <b>Dandan Qiao</b>, Jin Zhang, Guoqing Chen, Xunhua Guo. (2016). “A Novel Bipartite Graph based Competitiveness Degree Analysis from Query Logs”. <i>ACM Transactions on Knowledge Discovery from Data</i> (TKDD), 11(2), pp.1-25
      </>
    ),
    href: "https://dl.acm.org/doi/10.1145/2996196",
  },
];

export const conferencePublications: TimelinePublication[] = [
  {
    year: "2026",
    text: (
      <>
        Gang Li, <b>Dandan Qiao</b>, and Mingxuan Zheng. (2026). Structured event representation and stock return predictability. Conference presentation, <i>China International Conference in Finance</i> (CICF) 2026.
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Tong Yang, Junjing Huang, Xiaofan Li and <b>Dandan Qiao</b>, “AIGC on Marketing: A Theory-driven Design System and Empirical Evaluation”, <i>International Conference on Information Systems</i> (ICIS), Nashville, US, 2025.
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Bruce Yang, Junjing Huang, Xiaofan Li, <b>Dandan Qiao</b>, “An ADR Approach on a Multi-Agent System for Team Productivity and Coordination”, <i>International Conference on Information Systems</i> (ICIS), Nashville, US, 2025.
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Jingyuan Deng, Aditya Karanam, and <b>Dandan Qiao</b>, “Better Connection, More Innovation? Unveiling the Impact of Ride-Hailing Services on Local Knowledge Flows”, <i>International Conference on Information Systems</i> (ICIS), Nashville, US, 2025.
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Jingyuan Deng and <b>Dandan Qiao</b>, “Better Connection, More Innovation? Unveiling the Impact of Ride-Hailing Services on Local Knowledge Flows”, <i>The 85th Annual Meeting of the Academy of Management</i> (AOM), Copenhagen, Denmark, 2025
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Tong Yang, Junjing Huang, Xiaofan Li and <b>Dandan Qiao</b>, “AIGC on Marketing: A Theory-driven Design System and Empirical Evaluation”, <i>Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Paphos, Cyprus, 2025
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Jingyuan Deng and <b>Dandan Qiao</b>, “Better Connection, More Innovation? Unveiling the Impact of Ride-Hailing Services on Local Knowledge Flows”, <i>Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Paphos, Cyprus, 2025
      </>
    ),
  },
  {
    year: "2025",
    text: (
      <>
        Gang Li, <b>Dandan Qiao</b> and Mingxuan Zheng, “Enhancing Stock Return Prediction with LLM-Extracted News Events: Accurate and Interpretable Insights”, <i>Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Paphos, Cyprus, 2025
      </>
    ),
  },
  {
    year: "2024",
    text: (
      <>
        Haoyu Yuan, <b>Dandan Qiao</b>, Qiang Wei, “When Crowdfunding Meets InDemand: A Dynamic Analysis Using Hidden Markov Model, ” <i>International Conference on Information Systems</i> (ICIS), Bangkok, Thailand, 2024.
      </>
    ),
  },
  {
    year: "2024",
    text: (
      <>
        <b>Dandan Qiao</b>, Huaxia Rui, Qian Xiong, “AI and Freelancers: Has the Inflection Point Arrived?” <i>International Conference on Information Systems</i> (ICIS), Bangkok, Thailand, 2024.
      </>
    ),
  },
  {
    year: "2024",
    text: (
      <>
        Haoyu Yuan, <b>Dandan Qiao</b>, Qiang Wei, “When Crowdfunding Meets InDemand: A Dynamic Analysis Using Hidden Markov Model” <i>Twentieth Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Lisbon, Portugal, 2024.
      </>
    ),
  },
  {
    year: "2024",
    text: (
      <>
        Qian Xiong, <b>Dandan Qiao</b>, Jingjing Li, “Understanding Impact of Social Media Brand Activism on Product Sales” <i>Twentieth Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Lisbon, Portugal, 2024.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Dinghao Xi, Ying Lu, <b>Dandan Qiao</b>, Wei Xu, “Unveiling the Secrets of Collaboration on Video-Sharing Platforms” <i>International Conference on Information Systems</i> (ICIS), Hyderabad, India, 2023.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Dinghao Xi, Ying Lu, <b>Dandan Qiao</b>, Wei Xu, “Unveiling the Secrets of Collaboration on Video-Sharing Platforms” <i>INFORMS Conference on Information Systems and Technology</i> (CIST), Arizona, United States, 2023.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Haoyu Yuan, <b>Dandan Qiao</b>, Qiang Wei, “Decoding Persuasion: A Hierarchical Deep Learning Framework for Predicting Crowdfunding Success, ” <i>INFORMS Workshop on Data Science </i>(WDS), Arizona, United States, 2023.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Jingyuan Deng, <b>Dandan Qiao</b>, Warut Khern-Am-Nuai, “An Empirical Study about the Impact of Incentivized Reviews on Product Sales: The Case of Amazon Vine Program” The 16th China Summer Workshop on Information Management (CSWIM), Changsha, China, 2023.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Ying Lu, <b>Dandan Qiao</b>, Shu He, Xiaoying Xu, Bernard C Y Tan, “Exploration or Solidification: Modeling Users’ Dynamics of Collusive Behavior in Decentralized Online Communities” The 16th China Summer Workshop on Information Management (CSWIM), Changsha, China, 2023.
      </>
    ),
  },
  {
    year: "2023",
    text: (
      <>
        Tong Yang, Benjiang Lu, <b>Dandan Qiao</b>, “The Impact of Vehicle Purchase Tax: A Construal Level Analysis of Consumer Reviews” The annual Pacific Asia Conference on Information Systems (PACIS), Nanchang, China, 2023
      </>
    ),
  },
  {
    year: "2022",
    text: (
      <>
        Ying Lu, <b>Dandan Qiao</b>, Shu He, Bernard C Y Tan, “Secrets of Bellwethers: The Effect of Search Intermediary on Market Structure and Vendor Strategy in Darknet,” <i>INFORMS Conference on Information Systems and Technology</i> (CIST), Indianapolis, United States, 2022.
      </>
    ),
  },
  {
    year: "2022",
    text: (
      <>
        Jingyuan Deng, Grace Gu, <b>Dandan Qiao</b>, “Wisdom Overlooked: Spillover Effects of Firm-Driven Contest on User Innovation and Crowd Education,” <i>INFORMS Conference on Information Systems and Technology</i> (CIST), Indianapolis, United States, 2022.
      </>
    ),
  },
  {
    year: "2022",
    text: (
      <>
        Jingyuan Deng, Grace Gu, <b>Dandan Qiao</b>, “Whom to Listen to and When? Empirically Studying User Innovation over Product Development Life Cycle,” <i>Seventeenth Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Madrid, Spain, June 2022.
      </>
    ),
  },
  {
    year: "2022",
    text: (
      <>
        Jingyuan Deng, Grace Gu, <b>Dandan Qiao</b>, “Whom to Listen to and When? Empirically Studying User Innovation over Product Development Life Cycle,” <i>Strategic Management Society</i> (SMS), London, UK, 2022.
      </>
    ),
  },
  {
    year: "2020",
    text: (
      <>
        Lin Qiu, <b>Dandan Qiao</b>, Bernard C Y Tan, Andrew B. Whinston, “More than a Ride: An Investigation on Ride-Hailing Services and Racial Hate Crimes,” <i>Sixteenth Symposium on Statistical Challenges in Electronic Commerce Research</i> (SCECR), Madrid, Spain, June 2020.
      </>
    ),
  },
  {
    year: "2019",
    text: (
      <>
        Jason Chan, Shu He, <b>Dandan Qiao</b>, Andrew B. Whinston, “Shedding Light on the Dark: The Impact of Legal Enforcement on Darknet Transactions,” <i>Workshop on Information Systems and Economics</i> (WISE), Munich, Germany, December 2019.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, “Overcoming the Crowding-Out Effect of Monetary Incentive on Pro-Social Behavior,” <i>International Conference on Information Systems</i>, Seoul, South Korea, December 2017.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        <b>Dandan Qiao</b>, “Applying Financial Incentives in Online Pro-social Contexts: Crowdingout Effects and Overcoming Strategies,” <i>International Conference on Information Systems Doctoral Consortium</i>, Seoul, South Korea, December 2017.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, “Crowding-Out Effect Mitigation through Intervention Strategies,” <i>INFORMS Conference on Information Systems and Technology</i> (CIST), Houston, TX, October 2017.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        Samuel L. Blazek, <b>Dandan Qiao</b>, Andrew B. Whinston Guoqing Chen, “Dark Net Markets: Protect Your Privacy in Future E-commerce,” <i>Workshop on Information Systems and Economics</i> (WISE), Seoul, South Korea, December 2017.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        Samuel L. Blazek, <b>Dandan Qiao</b>, Deepayan Chakrabarti, Andrew B. Whinston, “The Price of Privacy: Insights from Anonymous Marketplace,” <i>Annual Digital Information Policy Scholars Conference</i>, George Mason University, Arlington, VA, April 2017.
      </>
    ),
  },
  {
    year: "2017",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, “Incentive Provision and Pro-Social Behaviors,” <i>Hawaii International Conference on System Sciences</i> (HICSS), Big Island, HI, January 2017.
      </>
    ),
  },
  {
    year: "2016",
    text: (
      <>
        <b>Dandan Qiao</b>, Shun-Yang Lee, Andrew B. Whinston, Qiang Wei, “Incentive Provision and Pro-Social Behaviors,” <i>INFORMS Conference on Information Systems and Technology</i> (CIST), Nashville, TN, November 2016.
      </>
    ),
  },
  {
    year: "2015",
    text: (
      <>
        <b>Dandan Qiao</b>, Jin Zhang, “A Novel Keyword Suggestion Method to Achieve Competitive Advertising on Search Engines,” <i>The Pacific Asia Conference on Information Systems</i> (PACIS), Singapore, July 2015.
      </>
    ),
  },
  {
    year: "2015",
    text: (
      <>
        <b>Dandan Qiao</b>, Qiang Wei, Jin Zhang, Guoqing Chen, “A Bipartite Graph Based Competitiveness Degrees Analysis with Query Logs on Search Engine,” <i>Conference of the International Fuzzy Systems Association and the European Society for Fuzzy Logic and Technology</i> (IFSA-EUSFLAT), Gij´on, Spain, June 2015.
      </>
    ),
  },
];

export const underReviewPublications: TimelinePublication[] = [
  /*
  {
    year: "",
    text: (
      <>
        Bin Li, Nakyung Kyung, <b>Dandan Qiao</b> and Luning Liu, “AI for VIP Customers: Examining Impact of AI Services on Customer Interactions” Under Review at <i>Production and Operations Management</i>.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        Duofeng Xu, Bryan Hooi, <b>Dandan Qiao</b>, “When Order Matters: First-Speaker Bias and Mitigation through Personality in Sequential Multi-Agent Debate”, Under Review at EMNLP (Empirical Methods in Natural Language Processing) 2026.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        Yingfei Shi, <b>Dandan Qiao</b>, Shukai Liu, “Your Like Is Not My Like: Propensity Aware Multi-Behavior Recommendation", Under Review at AAAI 2027.
      </>
    ),
  },
  */
];

export const workingPaperPublications: TimelinePublication[] = [
  {
    year: "",
    text: (
      <>
        “An Empirical Study about the Impact of Incentivized Reviews on Product Sales: The Case of Amazon Vine Program” with Jingyuan Deng, and Warut Khern-Am-Nuai, Working Ppaer.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Unraveling the Role of Public Crowds in Driving Innovation for Technology Product Development” with Jingyuan Deng, and Grace Gu, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Decoding Persuasion: A Hierarchical Deep Learning Framework for Predicting Crowdfunding Success”, with Haoyu Yuan, and Qiang Wei, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Understanding Impact of Social Media Brand Activism on Product Sales” with Qian Xiong, Mingxuan Zheng, and Jingjing Li, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Unveiling the Secrets of Collab oration in Online Video Sharing Platforms” with Dinghao Xi, Ying Lu, and Wei Xu, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Predictive Analytics in Changing Environments: Trade-Offs in Response to Changes in Data Patterns” with Jiaxu Peng, and Jungpil Hahn, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “Structured Event Representation and Stock Return Predictability” with Bing Han, Gang Li, and Mingxuan Zheng, Working Paper.
      </>
    ),
  },
  /*
  {
    year: "",
    text: (
      <>
        “Modelling Collusion Dynamics in Decentralized Online Social Media” with Ying Lu, Shu He, and Bernard C Y Tan, Working Paper.
      </>
    ),
  },
  {
    year: "",
    text: (
      <>
        “In Coherence We Trust: Designing an Innovative Review Ranking System” with Yiming Liu, Xiaofan Li, Working Paper.
      </>
    ),
  },
  */
];

export type StudentItem = {
  name: string;
  description: string;
  startYear: number;
  endYear?: number;
  picture?: string;
  pictureAlt?: string;
  profileUrl?: string;
};

export const alumniItems: StudentItem[] = [
  {
    name: "Ying Lu",
    description:
      "Former: PhD\nCurrent: Assistant Professor, IE University, Spain",
    startYear: 2019,
    endYear: 2025,
    picture: "alumni/ying-lu.webp",
    pictureAlt: "Ying Lu",
    profileUrl: "https://www.ie.edu/university/about/faculty/ying-lu/",
  },
  {
    name: "Haoyu Yuan",
    description:
      "Former: Visiting Student\nCurrent: Assistant Professor, Shanghai University of Finance and Economics",
    startYear: 2023,
    endYear: 2024,
    picture: "alumni/haoyu-yuan.png",
    pictureAlt: "Haoyu Yuan",
    profileUrl: "https://sime.sufe.edu.cn/99/3f/c10574a235839/page.htm",
  },
  {
    name: "Dinghao Xi",
    description:
      "Former: Visiting Student\nCurrent: Assistant Professor, Shanghai University of Finance and Economics",
    startYear: 2022,
    endYear: 2023,
    picture: "alumni/dinghao-xi.jpg",
    pictureAlt: "Dinghao Xi",
    profileUrl: "https://dinghaoxi.github.io/",
  },
  {
    name: "Qian Xiong",
    description: "Former: Visiting Student",
    startYear: 2022,
    endYear: 2023,
    picture: "alumni/qian-xiong.jpg",
    pictureAlt: "Qian Xiong",
    profileUrl: "https://www.linkedin.com/in/qian-xiong-128a921b3/",
  },
];

export const studentItems: StudentItem[] = [
  {
    name: "Yiming Liu",
    description: "PhD, NUS",
    startYear: 2021,
    picture: "students/yiming-liu.jpg",
    pictureAlt: "Yiming Liu",
  },
  {
    name: "Jingyuan Deng",
    description: "PhD, NUS",
    startYear: 2021,
    picture: "students/jingyuan-deng.avif",
    pictureAlt: "Jingyuan Deng",
    profileUrl: "https://sites.google.com/view/jingyuandeng/home/",
  },
  {
    name: "Dingyu Shi",
    description:
      "PhD Student\nResearch Interest: Digital Marketing, Econometrics",
    startYear: 2023,
    picture: "students/dingyu-shi.jpg",
    pictureAlt: "Dingyu Shi",
    profileUrl: "https://www.linkedin.com/in/dingyu-shi-02214331b/",
  },
  {
    name: "Yingfei Shi",
    description:
      "PhD Student\nResearch Interest: Recommendation, Computational Social Science",
    startYear: 2023,
    picture: "students/yingfei-shi.jpg",
    pictureAlt: "Yingfei Shi",
    profileUrl: "https://www.linkedin.com/in/yingfei-shi-4609a8195/",
  },
  {
    name: "Mingxuan Zheng",
    description:
      "PhD Student\nResearch Interest: Explainable AI, Fintech, LLM",
    startYear: 2023,
    picture: "students/mingxuan-zheng.jpg",
    pictureAlt: "Mingxuan Zheng",
    profileUrl: "https://www.linkedin.com/in/mingxuan-zheng-6a0613330/",
  },
  {
    name: "Duofeng Xu",
    description: "PhD, NUS",
    startYear: 2025,
    picture: "students/duofeng-xu.jpg",
    pictureAlt: "Duofeng Xu",
    profileUrl: "https://www.linkedin.com/in/xu-duofeng-72202716b/",
  },
  {
    name: "Wei Zhang",
    description: "PhD, NUS",
    startYear: 2025,
    picture: "students/wei-zhang.jpg",
    pictureAlt: "Wei Zhang",
    profileUrl: "https://www.linkedin.com/in/wei-zhang-a22802380/",
  },
];
